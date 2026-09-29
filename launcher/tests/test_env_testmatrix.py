"""--test compatibility matrix against a local mock endpoint."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from _helpers import mk_resolved
import ktl_env


class _Handler(BaseHTTPRequestHandler):
    # class-level flags the tests can toggle
    has_responses = True
    tools_supported = True

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "qwen3.8-27b"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        if self.path == "/v1/chat/completions":
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                return
            msg = {"role": "assistant", "content": "hi"}
            if body.get("tools") and type(self).tools_supported:
                msg["tool_calls"] = [{"function": {"name": "add", "arguments": "{}"}}]
            self._send(200, {"choices": [{"message": msg}]})
        elif self.path == "/v1/messages":
            self._send(200, {"type": "message",
                             "content": [{"type": "text", "text": "hi"}]})
        elif self.path == "/v1/responses":
            if type(self).has_responses:
                self._send(200, {"output": [{"type": "message"}], "status": "completed"})
            else:
                self._send(404, {"error": {"message": "not implemented"}})
        else:
            self._send(404, {"error": "not found"})


class TestMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _resolved(self):
        return mk_resolved(client="claude-code", base_url=f"http://127.0.0.1:{self.port}")

    def test_full_matrix_passes(self):
        _Handler.has_responses = True
        _Handler.tools_supported = True
        rows = ktl_env.run_matrix(self._resolved())
        by = {r["name"]: r for r in rows}
        for name in ("GET /v1/models",
                     "POST /v1/chat/completions (non-stream)",
                     "POST /v1/chat/completions (stream)",
                     "POST /v1/messages (Bearer)",
                     "POST /v1/messages (x-api-key)",
                     "POST /v1/responses (Codex)"):
            self.assertTrue(by[name]["ok"], f"{name} failed: {by[name]['detail']}")
        self.assertEqual(ktl_env.matrix_exit_code(rows), 0)

    def test_missing_responses_flags_failure(self):
        _Handler.has_responses = False
        try:
            rows = ktl_env.run_matrix(self._resolved())
            by = {r["name"]: r for r in rows}
            self.assertFalse(by["POST /v1/responses (Codex)"]["ok"])
            self.assertIn("NOT IMPLEMENTED", by["POST /v1/responses (Codex)"]["detail"])
            # other critical rows still pass -> exit 1 (some fail), not 2/3
            self.assertEqual(ktl_env.matrix_exit_code(rows), 1)
        finally:
            _Handler.has_responses = True

    def test_models_missing_served_model_is_a_failure(self):
        # a server that lists models but NOT the one we target must fail the
        # GET /v1/models row (an empty list is also a failure)
        class _Other(_Handler):
            def do_GET(self):
                self._send(200, {"data": [{"id": "some-other-model"}]})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Other)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = mk_resolved(client="claude-code", base_url=f"http://127.0.0.1:{port}")
            rows = ktl_env.run_matrix(r)
            by = {row["name"]: row for row in rows}
            self.assertFalse(by["GET /v1/models"]["ok"])
            self.assertIn("not in /v1/models", by["GET /v1/models"]["detail"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_unreachable_exit_2(self):
        r = mk_resolved(client="claude-code", base_url="http://127.0.0.1:1")  # closed port
        rows = ktl_env.run_matrix(r)
        self.assertEqual(ktl_env.matrix_exit_code(rows), 2)

    def test_auth_failure_exit_2(self):
        # point at the mock but with a key it would reject — simulate by using a
        # server that always 401s.
        class _Auth(_Handler):
            def do_GET(self):
                self._send(401, {"error": "unauthorized"})

            def do_POST(self):
                self._send(401, {"error": "unauthorized"})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Auth)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = mk_resolved(client="claude-code", base_url=f"http://127.0.0.1:{port}")
            rows = ktl_env.run_matrix(r)
            self.assertEqual(ktl_env.matrix_exit_code(rows), 2)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
