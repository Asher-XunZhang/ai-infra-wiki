"""Protocol tests use a local mock HTTP server, never a model or GPU."""
import contextlib
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location("serving_labs", Path(__file__).parents[1] / "scripts/serving_labs.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class Handler(BaseHTTPRequestHandler):
    aborted = threading.Event()
    controls = []

    def log_message(self, *_):
        pass

    def json_response(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.json_response(200, {"model_path": "protocol-test-only", "server": "mock"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path in ("/start_profile", "/stop_profile"):
            self.controls.append((self.path, payload))
            self.json_response(200, {"accepted": True})
            return
        if self.path == "/abort_request":
            self.aborted.set()
            self.json_response(200, {"notified": payload["rid"]})
            return
        if payload["text"] == "fail":
            self.json_response(503, {"error": "injected protocol failure"})
            return
        rid = payload["rid"]
        if not payload["stream"]:
            self.json_response(200, {"text": "协议样例", "meta_info": {
                "id": rid, "prompt_tokens": 7, "completion_tokens": 4,
                "finish_reason": {"type": "length"}}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.aborted.clear()

        def send(count, finish=None):
            value = {"text": "协议样例"[:count], "meta_info": {
                "id": rid, "prompt_tokens": 7, "completion_tokens": count,
                "finish_reason": {"type": finish} if finish else None}}
            self.wfile.write(("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()

        send(2)
        if payload["text"] == "cancel":
            if not self.aborted.wait(3):
                raise RuntimeError("cancel probe did not send its explicit-rid abort")
            send(2, "abort")
        else:
            send(4, "length")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(3)
        assert not cls.thread.is_alive()

    def invoke(self, action, directory, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return lab.main([action, "--url", self.url, "--output", str(directory), *args])

    def test_smoke_preserves_complete_and_streaming_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "smoke"
            self.assertEqual(self.invoke("smoke", dst), 0)
            run = json.loads((dst / "run.json").read_text())
            self.assertEqual(run["schema"], "ai-infra-serving-lab/v1")
            self.assertEqual(len(run["rows"]), 2)
            self.assertIsNone(run["rows"][0]["first"], "nonstreaming cannot prove first-token time")
            self.assertTrue(run["rows"][1]["done_marker"])
            self.assertEqual(len(run["rows"][1]["itls"]), 2,
                             "the first two-token chunk must not invent an internal interval")
            for row in run["rows"]:
                self.assertLessEqual(row["scheduled"], row["permit"])
                self.assertLessEqual(row["permit"], row["start"])
                self.assertLessEqual(row["start"], row["end"])
                self.assertEqual(row["out"], 4)
                self.assertTrue(row["success"])
                self.assertLessEqual(row["end"], run["duration"])
            self.assertIn(b"finish_reason", (dst / "request-0001.response.raw").read_bytes())
            self.assertFalse(run["manifest"]["server_commit_automatically_verified"])
            with self.assertRaises(FileExistsError):
                self.invoke("smoke", dst)

    def test_failure_stays_in_full_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.jsonl"
            src.write_text('\n'.join(json.dumps({"text": t, "max_new_tokens": 4}) for t in ["ok", "fail", "ok"]))
            dst = Path(tmp) / "bench"
            self.assertEqual(self.invoke("benchmark", dst, "--input", str(src), "--rate", "inf", "--concurrency", "1"), 1)
            run = json.loads((dst / "run.json").read_text())
            self.assertEqual(len(run["rows"]), 3)
            self.assertEqual(sum(r["success"] for r in run["rows"]), 2)
            self.assertIn("503", run["rows"][1]["error"])
            self.assertEqual(json.loads((dst / "request-0001.http.json").read_text())["status"], 503)
            self.assertEqual(len(run["manifest"]["input_sha256"]), 64)
            self.assertGreaterEqual(run["duration"], max(r["end"] for r in run["rows"]))

    def test_explicit_request_cancellation_and_profile_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "cancel"
            self.assertEqual(self.invoke("cancel", dst, "--prompt", "cancel"), 0)
            row = json.loads((dst / "run.json").read_text())["rows"][0]
            self.assertEqual(row["cancel_status"], 200)
            self.assertEqual(row["finish_reason"]["type"], "abort")
            self.assertFalse(row["success"], "intentional abort is not a completed benchmark request")
            self.assertEqual(self.invoke("profile-start", Path(tmp) / "start",
                                        "--server-output-dir", "./profiles/unique-run"), 0)
            self.assertEqual(self.invoke("profile-stop", Path(tmp) / "stop"), 0)
            start = next(p for route, p in Handler.controls if route == "/start_profile")
            self.assertEqual(start["activities"], ["CPU", "GPU"])
            self.assertEqual(start["output_dir"], "./profiles/unique-run")
            self.assertFalse(start["record_shapes"])
            self.assertFalse(start["with_stack"])
            self.assertNotIn("num_steps", start, "manual profile must not accidentally auto-stop")


if __name__ == "__main__":
    unittest.main()
