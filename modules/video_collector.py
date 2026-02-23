from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

from modules.config import get_config, set_config_path
from modules.video_backend import get_video_backend


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
	daemon_threads = True


class _State:
	def __init__(self) -> None:
		self.backend = get_video_backend()
		self.lock = threading.Lock()


STATE = _State()


class Handler(BaseHTTPRequestHandler):
	protocol_version = "HTTP/1.1"

	def _send_json(self, obj, status: int = 200) -> None:
		data = json.dumps(obj).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def do_GET(self) -> None:  # noqa: N802
		if self.path.startswith("/status"):
			st = STATE.backend.get_status()
			st["backend"] = STATE.backend.name()
			self._send_json(st)
			return

		if self.path.startswith("/snapshot.jpg"):
			jpeg, _t = STATE.backend.get_latest_jpeg()
			if jpeg is None:
				self.send_response(404)
				self.end_headers()
				return
			self.send_response(200)
			self.send_header("Content-Type", "image/jpeg")
			self.send_header("Content-Length", str(len(jpeg)))
			self.end_headers()
			self.wfile.write(jpeg)
			return

		if self.path.startswith("/mjpeg"):
			# parse fps query param (very lightweight parsing)
			fps = float(get_config().video.default_mjpeg_fps)
			try:
				if "fps=" in self.path:
					fps = float(self.path.split("fps=", 1)[1].split("&", 1)[0])
			except Exception:
				fps = float(get_config().video.default_mjpeg_fps)

			self.send_response(200)
			self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
			self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
			self.send_header("Pragma", "no-cache")
			self.send_header("Connection", "keep-alive")
			self.end_headers()

			min_interval = 1.0 / max(1.0, float(fps))
			last_t = None
			last_sent = 0.0
			try:
				while True:
					jpeg, t = STATE.backend.get_latest_jpeg()
					if jpeg is None or t is None:
						time.sleep(0.05)
						continue
					if last_t is not None and t == last_t:
						time.sleep(0.01)
						continue
					now = time.monotonic()
					if now - last_sent < min_interval:
						time.sleep(min_interval - (now - last_sent))
						continue

					last_t = t
					last_sent = time.monotonic()
					self.wfile.write(b"--frame\r\n")
					self.wfile.write(b"Content-Type: image/jpeg\r\n")
					self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
					self.wfile.write(jpeg + b"\r\n")
					try:
						self.wfile.flush()
					except Exception:
						break
			except Exception:
				return

		self.send_response(404)
		self.end_headers()

	def do_POST(self) -> None:  # noqa: N802
		if self.path.startswith("/connect"):
			try:
				STATE.backend.start()
			except Exception:
				pass
			self._send_json({"detail": "ok"})
			return

		if self.path.startswith("/disconnect"):
			try:
				STATE.backend.stop()
			except Exception:
				pass
			self._send_json({"detail": "ok"})
			return

		if self.path.startswith("/record/start"):
			# expected query params: session_dir and fps
			session_dir = ""
			fps = int(get_config().video.recording_fps)
			try:
				if "session_dir=" in self.path:
					session_dir = self.path.split("session_dir=", 1)[1].split("&", 1)[0]
					# naive URL decode for spaces only; good enough for local paths we generate
					session_dir = session_dir.replace("%2F", "/").replace("%5C", "\\").replace("%20", " ")
				if "fps=" in self.path:
					fps = int(self.path.split("fps=", 1)[1].split("&", 1)[0])
			except Exception:
				pass
			try:
				STATE.backend.start_recording(session_dir=session_dir, fps=int(fps))
				self._send_json({"detail": "ok"})
			except Exception as e:
				self._send_json({"detail": "error", "error": repr(e)}, status=500)
			return

		if self.path.startswith("/record/stop"):
			try:
				STATE.backend.stop_recording()
				self._send_json({"detail": "ok"})
			except Exception as e:
				self._send_json({"detail": "error", "error": repr(e)}, status=500)
			return

		self.send_response(404)
		self.end_headers()

	def log_message(self, fmt: str, *args) -> None:  # silence default logging
		return


def main(argv: Optional[list[str]] = None) -> int:
	cfg = get_config()
	p = argparse.ArgumentParser(description="Video collector process (optional)")
	p.add_argument("--config", default=None, help="Path to config.json (optional)")
	p.add_argument("--host", default=cfg.video.process.collector_host)
	p.add_argument("--port", type=int, default=int(cfg.video.process.collector_port))
	p.add_argument("--backend", default=None, help="Backend override (e.g. picamera2/jetson)")
	args = p.parse_args(argv)

	if args.config:
		try:
			set_config_path(args.config)
		except Exception:
			pass

	if args.backend:
		# Backend override is via CLI (not env) to keep config/env separation.
		STATE.backend = get_video_backend(backend_override=str(args.backend))

	srv = _ThreadingHTTPServer((args.host, int(args.port)), Handler)
	try:
		srv.serve_forever()
		return 0
	except KeyboardInterrupt:
		return 0
	finally:
		try:
			STATE.backend.stop()
		except Exception:
			pass


if __name__ == "__main__":
	raise SystemExit(main())


