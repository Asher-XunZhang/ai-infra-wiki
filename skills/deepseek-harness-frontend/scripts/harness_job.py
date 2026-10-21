#!/usr/bin/env python3
"""Launch guarded Harness batches; wait locally for milestones, never poll via an LLM."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
TERMINAL = {"completed", "checkpoint", "budget_blocked", "failed"}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def prepare(args):
    cwd = Path(args.cwd).resolve(strict=True)
    handoff = Path(args.handoff).resolve(strict=True)
    overlay = Path(args.overlay).resolve(strict=True)
    if not handoff.read_text().strip():
        raise ValueError("Empty handoff")
    dsh = shutil.which(args.dsh)
    if not dsh:
        raise ValueError("dsh not found")
    module = args.llm_module
    if module is None:
        # Desktop macOS layout; other installations supply --llm-module explicitly.
        resources = Path(dsh).resolve().parents[3]
        module = str(resources / "app.asar/dsh/node_modules/@deepseek-ai/dsh-llm/lib/index.js")
    if not 0 < args.warn_tokens < args.hard_tokens or min(args.max_responses, args.grace_responses) < 1:
        raise ValueError("Invalid budgets")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    os.chmod(run, 0o700)
    run_id = str(uuid.uuid4())
    config = {
        "runId": run_id, "cwd": str(cwd), "llmModule": module,
        "warnTokens": args.warn_tokens, "hardTokens": args.hard_tokens,
        "maxResponses": args.max_responses, "graceResponses": args.grace_responses,
        "stateFile": str(run / "guard.json"), "armedFile": str(run / "armed.json"),
        "checkpointFile": str(run / "delivery.json"), "requestFile": str(run / "checkpoint.request"),
    }
    write_json(run / "guard-overlay.json", [{"insert": [{
        "id": "ai-wiki-budget-guard", "name": str(HERE / "budget_guard.cjs"), "config": config,
    }]}])
    (run / "task.md").write_text(handoff.read_text())
    write_json(run / "job.json", {
        "run_id": run_id, "cwd": str(cwd), "dsh": dsh, "overlay": str(overlay),
    })
    return run


def run_job(run):
    job = read_json(run / "job.json")
    write_json(run / "runner.json", {"phase": "starting", "pid": os.getpid()})
    cmd = [job["dsh"], "--profile", "headless", "--patch", job["overlay"],
           "--patch", str(run / "guard-overlay.json"), "--json", "-"]
    process = None
    lock = None
    try:
        locks = Path(tempfile.gettempdir()) / "ai-wiki-harness-locks"
        locks.mkdir(mode=0o700, exist_ok=True)
        key = hashlib.sha256(job["cwd"].encode()).hexdigest()
        lock = (locks / key).open("a")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (run / "events.jsonl").open("wb") as out, (run / "stderr.log").open("wb") as err:
            process = subprocess.Popen(cmd, cwd=job["cwd"], stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, pass_fds=(lock.fileno(),))
            write_json(run / "runner.json", {
                "phase": "starting", "pid": os.getpid(), "harness_pid": process.pid,
            })
            deadline = time.monotonic() + 30
            while not (run / "armed.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Guard did not arm; task was not sent")
                time.sleep(0.1)
            armed = read_json(run / "armed.json")
            if armed.get("run_id") != job["run_id"] or armed.get("pid") != process.pid:
                raise RuntimeError("Guard identity mismatch; task was not sent")
            process.stdin.write((run / "task.md").read_bytes())
            process.stdin.close()
            write_json(run / "runner.json", {
                "phase": "running", "pid": os.getpid(), "harness_pid": process.pid,
            })
            code = process.wait()  # OS wait: no repeated model or shell calls.
        guard = read_json(run / "guard.json", {})
        phase = {"checkpoint_accepted": "checkpoint", "budget_blocked": "budget_blocked"}.get(
            guard.get("phase"), "completed" if code == 0 else "failed")
        if phase == "checkpoint" and not (run / "delivery.json").is_file():
            phase = "failed"
        write_json(run / "runner.json", {
            "phase": phase, "exit_code": code, "pid": os.getpid(),
            "harness_pid": process.pid, "guard": guard,
            "delivery": str(run / "delivery.json") if phase == "checkpoint" else None,
        })
    except Exception as exc:
        # Startup failed before handing off a task. Never force-kill an active editing run.
        if process and process.poll() is None and process.stdin and not process.stdin.closed:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()  # This exact child has not received a task.
                    process.wait(timeout=5)
        still_running = process is not None and process.poll() is None
        write_json(run / "runner.json", {
            "phase": "needs_attention" if still_running else "failed",
            "error": str(exc), "pid": os.getpid(), "process_finished": not still_running,
            "harness_pid": process.pid if process else None,
        })
        return 1
    finally:
        if process and process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if lock:
            lock.close()
    return 0


def snapshot(run):
    if not (run / "job.json").exists():
        raise ValueError("Not a guarded run directory")
    runner = read_json(run / "runner.json", {"phase": "starting"})
    guard = read_json(run / "guard.json", {})
    if runner["phase"] in TERMINAL or runner["phase"] == "needs_attention":
        return runner
    if guard.get("phase") in {"closing", "budget_blocked", "checkpoint_accepted"}:
        return {"phase": "closing", "guard": guard, "process_finished": False}
    return {"phase": runner["phase"]}


def watch(args):
    """One local watcher stays alive. Output only on budget/delivery/exit or a stall."""
    run = Path(args.run_dir).resolve()
    if args.stall_seconds <= 0 or args.timeout < 0:
        raise ValueError("Invalid wait interval")
    started = time.monotonic()
    last_progress = started
    previous_signature = None
    while True:
        if args.legacy_log:
            log, exit_file = Path(args.legacy_log), Path(args.legacy_exit)
            if exit_file.exists():
                code = int(exit_file.read_text().strip())
                result = {"phase": "completed" if code == 0 else "failed",
                          "exit_code": code, "legacy": True}
            else:
                result = {"phase": "running", "legacy": True}
            signature = log.stat().st_size if log.exists() else None
        else:
            result = snapshot(run)
            log = run / "events.jsonl"
            signature = log.stat().st_size if log.exists() else None
        if signature != previous_signature:
            previous_signature, last_progress = signature, time.monotonic()
        phase = result["phase"]
        if phase in TERMINAL or phase == "needs_attention" or (not args.terminal_only and phase in
                                 {"closing", "checkpoint_accepted"}):
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        if time.monotonic() - last_progress >= args.stall_seconds:
            print(json.dumps({"phase": "needs_attention", "reason": "no log progress",
                              "seconds": args.stall_seconds, "not_proof_of_failure": True}), flush=True)
            return 2
        if args.timeout and time.monotonic() - started >= args.timeout:
            return 124  # No status dump, no new model request.
        time.sleep(0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    for key in ("cwd", "handoff", "overlay", "run-dir"):
        start.add_argument("--" + key, required=True)
    start.add_argument("--dsh", default="dsh")
    start.add_argument("--llm-module")
    start.add_argument("--warn-tokens", type=int, default=128000)
    start.add_argument("--hard-tokens", type=int, default=256000)
    start.add_argument("--max-responses", type=int, default=40)
    start.add_argument("--grace-responses", type=int, default=8)
    worker = sub.add_parser("_run")
    worker.add_argument("--run-dir", required=True)
    waiting = sub.add_parser("watch")
    waiting.add_argument("--run-dir", required=True)
    waiting.add_argument("--terminal-only", action="store_true")
    waiting.add_argument("--timeout", type=float, default=0)
    waiting.add_argument("--stall-seconds", type=float, default=900)
    waiting.add_argument("--legacy-log")
    waiting.add_argument("--legacy-exit")
    for name in ("status", "request-checkpoint"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run-dir", required=True)
    args = p.parse_args()
    if args.command == "start":
        run = prepare(args)
        with (run / "worker.log").open("wb") as log:
            worker = subprocess.Popen([sys.executable, "-B", __file__, "_run", "--run-dir", str(run)],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                      start_new_session=True)
        print(json.dumps({"phase": "starting", "run_dir": str(run), "pid": worker.pid}))
    elif args.command == "_run":
        return run_job(Path(args.run_dir).resolve())
    elif args.command == "watch":
        if bool(args.legacy_log) != bool(args.legacy_exit):
            p.error("legacy-log and legacy-exit must be provided together")
        return watch(args)
    elif args.command == "status":
        print(json.dumps(snapshot(Path(args.run_dir).resolve()), ensure_ascii=False))
    else:
        run = Path(args.run_dir).resolve()
        if not (run / "job.json").exists():
            p.error("Not a guarded job; cannot inject a checkpoint request")
        (run / "checkpoint.request").touch()
        print(json.dumps({"phase": "checkpoint_requested", "applies_at": "next pre-step"}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"phase": "failed", "error": str(exc)}), file=sys.stderr)
        sys.exit(1)
