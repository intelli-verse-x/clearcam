"""Protocol/storage tests with synthetic files; no camera/model initialization."""
import ast
from contextlib import redirect_stdout
from email.parser import BytesParser
from email.policy import default
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.parse import urlparse, unquote, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.exports import exported_recording, recording_directory
from utils.notifications import send_notif, should_send_notification

ROOT = Path(__file__).resolve().parents[1]


class NotificationTests(unittest.TestCase):
    def message(self, host, image, token=None):
        connection = unittest.mock.MagicMock()
        response = connection.getresponse.return_value
        response.status, response.reason = 200, "OK"
        response.read.return_value = b"ok"
        with patch("http.client.HTTPSConnection", return_value=connection), patch("http.client.HTTPConnection", return_value=connection), redirect_stdout(io.StringIO()):
            send_notif(token, "synthetic title", "synthetic body", host=host, img=image)
        _, _, body, headers = connection.request.call_args.args
        message = BytesParser(policy=default).parsebytes(
            ("Content-Type: " + headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
        self.assertEqual(int(headers["Content-Length"]), len(body))
        boundary = message.get_boundary().encode()
        self.assertEqual(body.count(b"--" + boundary + b"--"), 1)
        self.assertTrue(body.endswith(b"--" + boundary + b"--\r\n"))
        return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                for part in message.iter_parts()}

    def test_custom_receiver_gets_image_before_closing_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "synthetic.jpg"
            image.write_bytes(b"synthetic image\x00\xff")
            parts = self.message("https://example.invalid", image)
        self.assertEqual(parts, {"text": b"synthetic title", "body_text": b"synthetic body", "img": b"synthetic image\x00\xff"})

    def test_default_receiver_keeps_existing_no_image_contract(self):
        parts = self.message("https://clearcam.org", "not-opened.jpg", "synthetic-token")
        self.assertEqual(parts["session_token"], b"synthetic-token")
        self.assertNotIn("img", parts)

    def test_disabled_notifications_and_vod_never_send(self):
        for enabled in (False, None, True, "synthetic-user"):
            for vod in (False, True):
                for notification in (False, True):
                    settings = types.SimpleNamespace(use_notifs=lambda: enabled)
                    self.assertEqual(should_send_notification(settings, vod, notification),
                                     bool(enabled and not vod and notification))

    def test_frame_processing_uses_the_tested_opt_out_guard(self):
        tree = ast.parse((ROOT / "clearcam.py").read_text())
        video = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VideoCapture")
        method = next(n for n in video.body if isinstance(n, ast.FunctionDef) and n.name == "process_frame")
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "should_send_notification"]
        self.assertEqual(len(calls), 1)


class InferenceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        # Importing helpers initializes ML dependencies. Compile only the reviewed
        # helper under test, with TinyJit replaced by an identity decorator.
        tree = ast.parse((ROOT / "utils/helpers.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "jit_infer")
        namespace = {"TinyJit": lambda fn: fn}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<isolated jit_infer>", "exec"), namespace)
        self.infer = namespace["jit_infer"]
        self.tensor = types.SimpleNamespace(shape=(2, 3))

    def test_original_positional_cache_and_new_keyword_cache(self):
        for call in (lambda fn, cache: self.infer(fn, self.tensor, cache),
                     lambda fn, cache: self.infer(fn, self.tensor, jit_cache=cache),
                     lambda fn, cache: self.infer(fn, x=self.tensor, jit_cache=cache)):
            cache = {}
            self.assertIs(call(lambda x: x, cache), self.tensor)
            self.assertIs(call(lambda x: x, cache), self.tensor)
            self.assertEqual(len(cache), 1)

    def test_multi_input_model_keeps_original_shape(self):
        self.assertEqual(self.infer(lambda tensor, shape: shape, self.tensor, (480, 640, 3), jit_cache={}), (480, 640, 3))

    def test_ambiguous_or_missing_cache_fails_clearly(self):
        with self.assertRaises(TypeError): self.infer(lambda x: x, self.tensor)
        with self.assertRaises(TypeError): self.infer(lambda x: x, self.tensor, x=self.tensor, jit_cache={})


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.stream = self.base / "cameras" / "front" / "streams" / "2026-11-12"
        self.stream.mkdir(parents=True)

    def request_export(self, path):
        # Exercise the actual handler method without importing its module, which
        # otherwise initializes camera storage, model state, and worker threads.
        tree = ast.parse((ROOT / "clearcam.py").read_text())
        handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HLSRequestHandler")
        method = next(n for n in handler.body if isinstance(n, ast.FunctionDef) and n.name == "do_GET")
        namespace = dict(urlparse=urlparse, unquote=unquote, parse_qs=parse_qs,
                         exported_recording=exported_recording, BASE_DIR=self.base,
                         shutil=shutil, subprocess=subprocess)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<isolated HTTP handler>", "exec"), namespace)
        request = types.SimpleNamespace(path=path, wfile=io.BytesIO(), send_response=unittest.mock.Mock(),
                                        send_error=unittest.mock.Mock(), send_header=unittest.mock.Mock(),
                                        end_headers=unittest.mock.Mock())
        namespace["do_GET"](request)
        return request

    def test_http_route_rejects_encoded_traversal_and_missing_recordings(self):
        for path in ("/export?cam=..&date=x", "/export?cam=front&date=..%2F..%2Fprivate", "/export?cam=front&date=%252Ftmp"):
            request = self.request_export(path)
            self.assertEqual(request.send_error.call_args.args[0], 400)
            request.send_response.assert_not_called()
        request = self.request_export("/export?cam=missing&date=x")
        self.assertEqual(request.send_error.call_args.args[0], 404)
        request.send_response.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "native media tools are required for integration")
    def test_http_failed_native_export_returns_error_without_success_headers(self):
        (self.stream / "init.mp4").write_bytes(b"invalid synthetic recording")
        (self.stream / "stream_000001.m4s").write_bytes(b"invalid synthetic fragment")
        request = self.request_export("/export?cam=front&date=2026-11-12")
        self.assertEqual(request.send_error.call_args.args[0], 502)
        request.send_response.assert_not_called()
        self.assertFalse(list(self.stream.glob(".export-*")))

    def test_rejects_traversal_absolute_missing_and_symlink_escape(self):
        for camera, date in [("..", "2026-11-12"), ("/tmp", "x"), ("front", "../../x"), ("front", "a\\b"), (None, "x"), ("front", "")]:
            with self.assertRaises(ValueError): recording_directory(self.base, camera, date)
        with self.assertRaises(FileNotFoundError): recording_directory(self.base, "missing", "x")
        (self.stream.parent / "escape").symlink_to(self.base)
        with self.assertRaises(ValueError): recording_directory(self.base, "front", "escape")
        self.assertEqual(recording_directory(self.base, "front", "2026-11-12"), self.stream.resolve())

    def test_missing_fragments_fail_without_starting_ffmpeg(self):
        with patch("utils.exports.subprocess.Popen") as popen:
            with self.assertRaises(FileNotFoundError):
                with exported_recording(self.base, "front", "2026-11-12"): pass
        popen.assert_not_called()

    def test_symlink_fragment_is_rejected(self):
        (self.stream / "init.mp4").write_bytes(b"synthetic")
        (self.stream / "stream_000001.m4s").symlink_to(self.base / "private")
        with self.assertRaises(ValueError):
            with exported_recording(self.base, "front", "2026-11-12"): pass

    @unittest.skipUnless(shutil.which("ffmpeg"), "native media tools are required for integration")
    def test_failed_native_export_never_yields_output_and_cleans_up(self):
        (self.stream / "init.mp4").write_bytes(b"invalid synthetic recording")
        (self.stream / "stream_000001.m4s").write_bytes(b"invalid synthetic fragment")
        with self.assertRaises(RuntimeError):
            with exported_recording(self.base, "front", "2026-11-12"):
                self.fail("Failed conversion must not yield a successful HTTP body")
        self.assertFalse(list(self.stream.glob(".export-*")))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "native media tools are required for integration")
    def test_real_synthetic_fmp4_export_concurrency_and_cleanup(self):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=4",
                        "-t", "2", "-c:v", "libx264", "-g", "4", "-f", "hls", "-hls_time", "1",
                        "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
                        "-hls_segment_filename", str(self.stream / "stream_%06d.m4s"), str(self.stream / "stream.m3u8")], check=True)
        with exported_recording(self.base, "front", "2026-11-12") as first:
            with exported_recording(self.base, "front", "2026-11-12") as second:
                self.assertNotEqual(first, second)
                data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name", "-of", "json", str(second)]))
                self.assertEqual(data["streams"][0]["codec_name"], "h264")
                self.assertGreaterEqual(float(data["format"]["duration"]), 1.9)
            self.assertFalse(second.exists())
            self.assertTrue(first.exists())
        self.assertFalse(first.exists())
        self.assertFalse(list(self.stream.glob(".export-*")))


if __name__ == "__main__":
    unittest.main()
