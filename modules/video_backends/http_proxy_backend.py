from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from modules.config import get_config
from modules.video_backend import VideoBackend
from modules.video_tools import cut_mp4_clip_copy, mux_h264_to_mp4_async


class HttpProxyVideoBackend(VideoBackend):
	"""
	Backend that proxies to a local video_collector process over HTTP.
	This keeps Jetson-specific camera code out-of-process and out of the main server.
	"""

	def __init__(self, base_url: str) -> None:
		self._base = base_url.rstrip("/")
		self._cfg = get_config().http_proxy

	def name(self) -> str:
		# The collector knows the actual backend; expose as proxy.
		return "proxy"

	def start(self) -> None:
		self._post("/connect")

	def stop(self) -> None:
		self._post("/disconnect")

	def get_status(self) -> Dict[str, Any]:
		try:
			with urllib.request.urlopen(self._base + "/status", timeout=float(self._cfg.status_timeout_seconds)) as resp:
				return json.loads(resp.read().decode("utf-8"))
		except Exception as e:
			return {"running": False, "error": repr(e)}

	def start_recording(self, session_dir: str, fps: int) -> None:
		self._post(f"/record/start?session_dir={urllib.parse.quote(session_dir)}&fps={int(fps)}")

	def stop_recording(self) -> None:
		self._post("/record/stop")

	def get_latest_jpeg(self) -> tuple[Optional[bytes], Optional[float]]:
		# Not used for proxy mode; server uses mjpeg_stream/snapshot_jpeg.
		return None, None

	def get_frames_since(self, since_t_host: float, include_frame_idx: bool = True, include_width_height: bool = True) -> list[Dict[str, Any]]:
		# Proxy mode: frames are managed by the remote collector process
		# Server would need to query via HTTP if needed; for now return empty
		return []

	async def mjpeg_stream(self, fps: float) -> AsyncIterator[bytes]:
		url = self._base + f"/mjpeg?fps={float(fps)}"
		loop = asyncio.get_running_loop()

		def _open():
			return urllib.request.urlopen(url, timeout=float(self._cfg.mjpeg_open_timeout_seconds))

		resp = await loop.run_in_executor(None, _open)
		try:
			while True:
				chunk = await loop.run_in_executor(None, resp.read, int(self._cfg.mjpeg_read_chunk_bytes))
				if not chunk:
					break
				yield chunk
		finally:
			try:
				resp.close()
			except Exception:
				pass

	async def snapshot_jpeg(self) -> Optional[bytes]:
		url = self._base + "/snapshot.jpg"
		loop = asyncio.get_running_loop()

		def _fetch():
			with urllib.request.urlopen(url, timeout=float(self._cfg.snapshot_timeout_seconds)) as resp:
				return resp.read()

		try:
			return await loop.run_in_executor(None, _fetch)
		except Exception:
			return None

	def mux_to_mp4_best_effort(self, session_dir: Path, fps: int) -> bool:
		# Containerization is a file operation; we keep this in-process for now.
		h264 = session_dir / "video.h264"
		mp4 = session_dir / "video.mp4"
		return mux_h264_to_mp4_async(h264, mp4, fps=int(fps))

	def cut_clip_best_effort(self, mp4_path: Path, out_path: Path, start_sec: float, duration_sec: float) -> bool:
		return cut_mp4_clip_copy(mp4_path, out_path, start_sec=float(start_sec), duration_sec=float(duration_sec))

	def _post(self, path: str) -> None:
		try:
			req = urllib.request.Request(self._base + path, method="POST", data=b"")
			with urllib.request.urlopen(req, timeout=float(self._cfg.post_timeout_seconds)) as _:
				return
		except Exception:
			return


