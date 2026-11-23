#!/usr/bin/env python3
"""Collect native SGLang HTTP evidence for the Pages serving labs.

Uses only the Python standard library. It never installs or starts a model.
Each invocation creates a new output directory and preserves raw responses.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


SCHEMA = "ai-infra-serving-lab/v1"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def endpoint(base, path):
    parts = urllib.parse.urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("--url must be an HTTP(S) service base URL without a query or fragment")
    return base.rstrip("/") + "/" + path.lstrip("/")


def exchange(base, path, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(endpoint(base, path), data=data,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return response.status, dict(response.headers), body
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def identity(base, directory, timeout):
    result = {}
    for name in ("model_info", "server_info"):
        status, headers, body = exchange(base, name, None, timeout)
        (directory / (name + ".raw")).write_bytes(body)
        write_json(directory / (name + ".http.json"), {"status": status, "headers": headers})
        try:
            value = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            value = None
        result[name] = {"status": status, "body": value}
        if status != 200 or value is None:
            raise ValueError(name + " did not return a readable HTTP 200 response; raw evidence retained")
    return result


def request_row(base, prompt, count, index, directory, clock0, scheduled, permit, timeout,
                stream=True, cancel=False):
    rid = "wiki-lab-" + uuid.uuid4().hex
    payload = {"rid": rid, "text": prompt,
               "sampling_params": {"temperature": 0, "max_new_tokens": count},
               "stream": stream}
    prefix = directory / ("request-" + str(index).zfill(4))
    write_json(prefix.with_suffix(".request.json"), payload)
    encoded = json.dumps(payload).encode()
    row = {"id": rid, "index": index, "scheduled": scheduled, "permit": permit,
           "start": None, "first": None, "end": None, "out": 0, "input": None,
           "success": False, "error": "", "finish_reason": None, "stream": stream,
           "itls": [], "chunks": []}
    raw = bytearray()
    ended = False
    cumulative = 0
    previous_chunk = None
    cancel_sent = False
    service_error = None
    http_status = None
    headers = {}

    def observe(obj, observed_at):
        nonlocal ended, cumulative, previous_chunk, cancel_sent, service_error
        if not isinstance(obj, dict):
            service_error = "response event is not a JSON object"
            return
        if obj.get("error"):
            service_error = obj["error"]
        meta = obj.get("meta_info") or {}
        if not isinstance(meta, dict):
            service_error = "meta_info is not an object"
            return
        if meta.get("id"):
            row["id"] = str(meta["id"])
        if isinstance(meta.get("prompt_tokens"), int):
            row["input"] = meta["prompt_tokens"]
        n = meta.get("completion_tokens")
        first_chunk = stream and bool(obj.get("text")) and row["first"] is None
        if first_chunk:
            row["first"] = observed_at
        if isinstance(n, int) and not isinstance(n, bool) and n >= cumulative:
            delta = n - cumulative
            if stream and obj.get("text"):
                if not first_chunk and previous_chunk is not None and delta > 0:
                    # Same native-adapter approximation: distribute the observed
                    # chunk interval over the new cumulative output count.
                    row["itls"].extend([(observed_at - previous_chunk) / delta] * delta)
            cumulative = n
        elif n is not None:
            service_error = "invalid or decreasing cumulative completion_tokens"
        if stream and obj.get("text"):
            previous_chunk = observed_at
        row["chunks"].append({"observed": observed_at, "completion_tokens": n,
                              "has_text": bool(obj.get("text"))})
        if meta.get("finish_reason") is not None:
            row["finish_reason"] = meta["finish_reason"]
            ended = True
        if cancel and not cancel_sent and row["first"] is not None:
            # An explicit rid avoids aborting unrelated requests. A successful
            # notification does not imply device resources have retired.
            cancel_sent = True
            status, ch, body = exchange(base, "abort_request",
                                       {"rid": row["id"], "abort_all": False}, timeout)
            prefix.with_suffix(".cancel.raw").write_bytes(body)
            write_json(prefix.with_suffix(".cancel.http.json"),
                       {"status": status, "headers": ch, "rid": row["id"]})
            row["cancel_status"] = status

    try:
        req = urllib.request.Request(endpoint(base, "generate"), data=encoded,
                                     headers={"Content-Type": "application/json"})
        row["start"] = time.perf_counter() - clock0
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            http_status, headers = exc.code, dict(exc.headers)
            raw.extend(exc.read())
            raise ValueError("HTTP " + str(exc.code)) from exc
        with response:
            http_status, headers = response.status, dict(response.headers)
            if stream:
                event_lines = []

                def consume():
                    if not event_lines:
                        return
                    data = "\n".join(event_lines)
                    if data.strip() == "[DONE]":
                        row["done_marker"] = True
                    else:
                        observe(json.loads(data), time.perf_counter() - clock0)

                for line in response:
                    raw.extend(line)
                    text = line.decode("utf-8").rstrip("\r\n")
                    if not text:
                        consume()
                        event_lines = []
                    elif text.startswith("data:"):
                        event_lines.append(text[5:].lstrip(" "))
                consume()
            else:
                raw.extend(response.read())
                observe(json.loads(raw), time.perf_counter() - clock0)
        row["out"] = cumulative
        if service_error:
            raise ValueError("service error: " + str(service_error))
        if not ended:
            raise ValueError("missing finish_reason; a partial text or [DONE] alone is insufficient")
        if not cumulative:
            raise ValueError("missing positive completion_tokens; raw evidence retained")
        reason = row["finish_reason"]
        reason_type = str(reason.get("type", "")) if isinstance(reason, dict) else str(reason)
        row["success"] = reason_type in ("stop", "length")
        if not row["success"]:
            row["error"] = "finish_reason=" + reason_type
    except Exception as exc:
        row["error"] = str(exc)
        row["out"] = cumulative
    finally:
        if row["start"] is None:
            row["start"] = time.perf_counter() - clock0
        row["end"] = time.perf_counter() - clock0
        row["cancel_sent"] = cancel_sent
        prefix.with_suffix(".response.raw").write_bytes(raw)
        write_json(prefix.with_suffix(".http.json"), {"status": http_status, "headers": headers})
        write_json(prefix.with_suffix(".observations.json"), row)
    return row


def load_requests(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("request file is empty")
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("each request needs nonempty text")
        n = row.get("max_new_tokens")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 16384:
            raise ValueError("max_new_tokens must be an integer from 1 to 16384")
    return rows, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "cancel", "benchmark", "profile-start", "profile-stop"])
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True, type=Path, help="a new evidence directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--input", type=Path, help="JSONL with text and max_new_tokens")
    parser.add_argument("--rate", type=float, default=float("inf"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--server-output-dir", help="profile directory interpreted by the service process")
    parser.add_argument("--source-commit", help="operator-declared server commit, not automatically verified")
    args = parser.parse_args(argv)
    endpoint(args.url, "model_info")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be finite and positive")
    if not 1 <= args.concurrency <= 256 or not 1 <= args.max_new_tokens <= 16384:
        parser.error("invalid concurrency or output budget")
    if args.rate <= 0 or math.isnan(args.rate):
        parser.error("--rate must be positive, or inf")
    if args.action == "benchmark" and not args.input:
        parser.error("benchmark requires --input")
    if args.action == "profile-start" and not args.server_output_dir:
        parser.error("profile-start requires --server-output-dir")
    # Refuse reuse instead of overwriting observations from an earlier run.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "action": args.action, "url": args.url,
                "created_unix": time.time(), "declared_server_commit": args.source_commit,
                "server_commit_automatically_verified": False,
                "client": "Python stdlib urllib; monotonic perf_counter",
                "timeout_seconds": args.timeout, "args": {k: str(v) if isinstance(v, Path) else
                    ("inf" if isinstance(v, float) and math.isinf(v) else v)
                    for k, v in vars(args).items()}}
    write_json(args.output / "manifest.json", manifest)
    if args.action.startswith("profile-"):
        body = {"output_dir": args.server_output_dir, "activities": ["CPU", "GPU"],
                "profile_by_stage": False, "with_stack": False, "record_shapes": False,
                "merge_profiles": False, "profile_id": "wiki-lab-" + uuid.uuid4().hex}
        status, headers, response = exchange(args.url, "start_profile" if args.action == "profile-start"
                                             else "stop_profile", body if args.action == "profile-start" else {}, args.timeout)
        (args.output / "control.raw").write_bytes(response)
        write_json(args.output / "control.http.json", {"status": status, "headers": headers,
                   "request": body if args.action == "profile-start" else {}})
        print("Control HTTP", status, "; evidence:", args.output,
              "; verify service-side logs and per-rank files separately.")
        return 0 if status == 200 else 1
    manifest["service"] = identity(args.url, args.output, args.timeout)
    write_json(args.output / "manifest.json", manifest)
    if args.action == "benchmark":
        requests, digest = load_requests(args.input)
        manifest["input_sha256"] = digest
        manifest["requests"] = len(requests)
        write_json(args.output / "manifest.json", manifest)
        generator = random.Random(args.seed)
        schedule, stamp = [], 0.0
        for _ in requests:
            schedule.append(stamp)
            if not math.isinf(args.rate):
                stamp += generator.expovariate(args.rate)
        write_json(args.output / "arrival-plan.json", schedule)
        semaphore = threading.BoundedSemaphore(args.concurrency)
        # A dispatcher does not wait for a permit before scheduling the next
        # arrival. Each task records planned arrival, permit and adapter start.
        clock0 = time.perf_counter()

        def task(i):
            wait = schedule[i] - (time.perf_counter() - clock0)
            if wait > 0:
                time.sleep(wait)
            with semaphore:
                permit = time.perf_counter() - clock0
                return request_row(args.url, requests[i]["text"], requests[i]["max_new_tokens"],
                                   i, args.output, clock0, schedule[i], permit, args.timeout)

        # Bounded workers avoid one OS thread per request. If the dispatcher
        # pool becomes a bottleneck, scheduled→permit explicitly retains it.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(requests), max(32, args.concurrency * 2))) as pool:
            rows = list(pool.map(task, range(len(requests))))
    else:
        clock0 = time.perf_counter()
        modes = [False, True] if args.action == "smoke" else [True]
        rows = []
        for i, stream in enumerate(modes):
            at = time.perf_counter() - clock0
            rows.append(request_row(args.url, args.prompt, args.max_new_tokens, i,
                                    args.output, clock0, at, at, args.timeout,
                                    stream=stream, cancel=args.action == "cancel"))
    duration = time.perf_counter() - clock0
    run = {"schema": SCHEMA, "evidence": "measured", "unit": "s", "action": args.action,
           "duration": duration, "rows": rows, "manifest": manifest,
           "window": "before dispatch to all tasks finished; excludes earlier identity checks",
           "ttft": "adapter start to first nonempty text observation; non-streaming first is unknown",
           "itl": "native cumulative chunk approximation; not individual GPU token timestamps"}
    write_json(args.output / "run.json", run)
    completed = sum(r["success"] for r in rows)
    print("Completed:", completed, "/", len(rows), "; evidence:", args.output / "run.json")
    if args.action == "cancel":
        return 0 if all(r.get("cancel_status") == 200 and r["finish_reason"] is not None for r in rows) else 1
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
