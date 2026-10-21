"""Offline lifecycle tests: fake CLI, real pipes/processes, no provider calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("harness_job.py")
spec = importlib.util.spec_from_file_location("harness_job", SCRIPT)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="harness-job-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / "task").write_text("Keep the existing files.")
        (self.base / "overlay").write_text("[]")

    def prepare(self, body):
        cli = self.base / "dsh"
        cli.write_text("#!" + sys.executable + "\n" + body)
        cli.chmod(0o700)
        return job.prepare(argparse.Namespace(
            cwd=str(self.base), handoff=str(self.base / "task"),
            overlay=str(self.base / "overlay"), run_dir=str(self.base / "run"),
            dsh=str(cli), llm_module="fake", warn_tokens=100, hard_tokens=200,
            max_responses=2, grace_responses=2))

    def test_guarded_checkpoint_process(self):
        run = self.prepare("""
import json,os,pathlib,sys
c=json.loads(pathlib.Path(sys.argv[-3]).read_text())[0]['insert'][0]['config']
pathlib.Path(c['armedFile']).write_text(json.dumps({'run_id':c['runId'],'pid':os.getpid()}))
task=sys.stdin.read()
pathlib.Path(c['cwd'],'preserved.txt').write_text(task)
pathlib.Path(c['checkpointFile']).write_text(json.dumps({'unit':'finished unit'}))
pathlib.Path(c['stateFile']).write_text(json.dumps({'phase':'checkpoint_accepted'}))
sys.exit(1)
""")
        self.assertEqual(job.run_job(run), 0)
        self.assertEqual(job.snapshot(run)["phase"], "checkpoint")
        self.assertEqual((self.base / "preserved.txt").read_text(), "Keep the existing files.")

    def test_missing_guard_never_sends_task(self):
        run = self.prepare("import sys; sys.exit(1)\n")
        self.assertEqual(job.run_job(run), 1)
        self.assertIn("not sent", job.snapshot(run)["error"])

    def test_unknown_run_is_not_reported_running(self):
        with self.assertRaises(ValueError):
            job.snapshot(self.base)

    def test_guard_reject_is_not_process_exit(self):
        run = self.prepare("import sys; sys.exit(1)\n")
        job.write_json(run / "runner.json", {"phase": "running"})
        job.write_json(run / "guard.json", {"phase": "budget_blocked"})
        self.assertEqual(job.snapshot(run)["phase"], "closing")
        self.assertFalse(job.snapshot(run)["process_finished"])

    def test_watcher_is_silent_until_completion(self):
        run = self.prepare("import sys; sys.exit(1)\n")
        job.write_json(run / "runner.json", {"phase": "running"})
        p = subprocess.Popen([sys.executable, "-B", str(SCRIPT), "watch",
                              "--run-dir", str(run), "--timeout", "4"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                p.communicate(timeout=0.2)
            job.write_json(run / "runner.json", {"phase": "completed", "exit_code": 0})
            out, err = p.communicate(timeout=3)
            self.assertEqual(json.loads(out)["phase"], "completed")
            self.assertEqual(err, "")
        finally:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=3)

    def test_legacy_watch_and_quiet_timeout(self):
        log, exit_file = self.base / "old.jsonl", self.base / "old.exit"
        log.write_text("{}\n")
        cmd = [sys.executable, "-B", str(SCRIPT), "watch", "--run-dir", str(self.base),
               "--legacy-log", str(log), "--legacy-exit", str(exit_file), "--timeout", "0.1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        self.assertEqual(r.returncode, 124)
        self.assertEqual(r.stdout, "")
        exit_file.write_text("0")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        self.assertTrue(json.loads(r.stdout)["legacy"])


if __name__ == "__main__":
    unittest.main()
