from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules import db


def _safe_read_json(path: Path) -> Dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


async def _wait_for_file(path: Path, timeout_s: float) -> bool:
	deadline = time.time() + float(timeout_s)
	while time.time() < deadline:
		if path.exists() and path.stat().st_size > 0:
			return True
		await asyncio.sleep(0.5)
	return bool(path.exists() and path.stat().st_size > 0)


async def _process_job(job: Dict[str, Any]) -> str:
	"""
	Job fields:
	- jump_id: int (required)
	- event_id: int (optional, used for naming)
	- session_id: str (required)
	- clip_start_host: float (required)
	- clip_end_host: float (required)
	- video_fps: int (optional, default 30)
	"""
	jump_id = int(job.get("jump_id") or 0)
	session_id = str(job.get("session_id") or "").strip()
	if jump_id <= 0 or not session_id:
		return "invalid job (missing jump_id/session_id)"

	clip_start_host = float(job.get("clip_start_host"))
	clip_end_host = float(job.get("clip_end_host"))
	video_fps = int(job.get("video_fps") or 30)

	base = Path("data") / "sessions" / session_id
	mp4 = base / "video.mp4"
	h264 = base / "video.h264"

	# Wait for MP4; if missing, we can't reliably clip for browser playback.
	# (We enqueue on detection; processing may happen after session stop/mux.)
	ok = await _wait_for_file(mp4, timeout_s=float(job.get("wait_mp4_timeout_s") or 900))
	if not ok:
		# If mp4 isn't ready but h264 exists, leave the job pending; caller can retry.
		if h264.exists():
			return "mp4 not ready yet (h264 exists) — retry later"
		return "mp4 not found"

	# Load session start host time from DB frames (preferred) or frames.csv
	start_host: Optional[float] = None
	try:
		fs = await db.get_frames(session_id=session_id, limit=1)
		if fs and isinstance(fs[0].get("t_host"), (int, float)):
			start_host = float(fs[0]["t_host"])
	except Exception:
		start_host = None
	if start_host is None:
		p = base / "frames.csv"
		if p.exists():
			try:
				with open(p, "r", encoding="utf-8") as fh:
					_ = fh.readline()
					line = fh.readline()
				parts = (line or "").strip().split(",")
				if len(parts) >= 2 and parts[1]:
					start_host = float(parts[1])
			except Exception:
				start_host = None
	if start_host is None:
		return "no session start_host available (frames missing)"

	start_sec = max(0.0, clip_start_host - start_host)
	duration = max(0.2, clip_end_host - clip_start_host)

	ffmpeg = shutil.which("ffmpeg")
	if not ffmpeg:
		return "ffmpeg not installed"

	clips_dir = base / "clips"
	clips_dir.mkdir(parents=True, exist_ok=True)
	event_id = int(job.get("event_id") or 0)
	out_name = f"jump_{event_id}.mp4" if event_id > 0 else f"jump_{jump_id}.mp4"
	out_path = clips_dir / out_name

	# Cut clip (fast, stream copy)
	subprocess.run(
		[
			ffmpeg,
			"-y",
			"-ss",
			f"{start_sec:.3f}",
			"-t",
			f"{duration:.3f}",
			"-i",
			str(mp4),
			"-c",
			"copy",
			"-movflags",
			"+faststart",
			str(out_path),
		],
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		check=False,
	)
	if not out_path.exists() or out_path.stat().st_size <= 0:
		return "ffmpeg clip failed"

	# Update jump video_path using jump_id (preferred)
	rel = str(Path("data") / "sessions" / session_id / "clips" / out_name)
	await db.set_jump_video_path_by_jump_id(jump_id, rel)

	# Persist per-jump clip-relative frames (jump_frames) for sync
	try:
		session_frames = await db.get_frames(session_id=session_id, t0=clip_start_host, t1=clip_end_host, limit=500000)
	except Exception:
		session_frames = []
	if not session_frames:
		# fallback: parse frames.csv and filter
		p = base / "frames.csv"
		if p.exists():
			try:
				tmp = []
				with open(p, "r", encoding="utf-8") as fh:
					_ = fh.readline()
					for line in fh:
						parts = line.strip().split(",")
						if len(parts) < 5:
							continue
						try:
							th = float(parts[1])
						except Exception:
							continue
						if th < clip_start_host or th > clip_end_host:
							continue
						tmp.append(
							{
								"frame_idx": int(parts[0]),
								"t_host": th,
								"device_ts": float(parts[2]) if parts[2] else None,
								"width": int(parts[3]) if parts[3] else None,
								"height": int(parts[4]) if parts[4] else None,
							}
						)
				session_frames = tmp
			except Exception:
				session_frames = []

	jf = []
	for i, f in enumerate(session_frames):
		try:
			th = float(f.get("t_host"))
		except Exception:
			continue
		jf.append(
			{
				"frame_idx": i,
				"t_video": max(0.0, th - clip_start_host),
				"t_host": th,
				"device_ts": f.get("device_ts"),
				"width": f.get("width"),
				"height": f.get("height"),
			}
		)
	await db.replace_jump_frames(jump_id, jf)

	return f"ok (clip={rel}, frames={len(jf)})"


async def _worker_loop(jobs_dir: Path, poll_s: float) -> None:
	# Ensure DB initialized if configured
	try:
		await db.init_db()
	except Exception:
		# DB optional; worker becomes a no-op
		pass

	done_dir = jobs_dir / "done"
	fail_dir = jobs_dir / "failed"
	jobs_dir.mkdir(parents=True, exist_ok=True)
	done_dir.mkdir(parents=True, exist_ok=True)
	fail_dir.mkdir(parents=True, exist_ok=True)

	while True:
		try:
			for path in sorted(jobs_dir.glob("*.json")):
				processing = path.with_suffix(".processing")
				try:
					path.rename(processing)
				except Exception:
					continue
				job = _safe_read_json(processing)
				try:
					msg = await _process_job(job)
					if msg.startswith("ok"):
						processing.rename(done_dir / processing.name.replace(".processing", ".done.json"))
					else:
						# Put back into queue for retry if mp4 not ready; otherwise mark failed.
						if "retry later" in msg:
							processing.rename(path)  # back to pending
						else:
							processing.rename(fail_dir / processing.name.replace(".processing", ".failed.json"))
							_write_text(fail_dir / (processing.stem + ".error.txt"), msg)
				except Exception as e:
					try:
						processing.rename(fail_dir / processing.name.replace(".processing", ".failed.json"))
						_write_text(fail_dir / (processing.stem + ".error.txt"), repr(e))
					except Exception:
						pass
		except Exception:
			pass
		await asyncio.sleep(float(poll_s))


def main(argv: Optional[list[str]] = None) -> int:
	p = argparse.ArgumentParser(description="Jump clip worker (runs out-of-process)")
	p.add_argument("--jobs-dir", default="data/jobs/jump_clips", help="Directory containing queued job JSON files")
	p.add_argument("--poll-s", type=float, default=0.5, help="Polling interval seconds")
	args = p.parse_args(argv)

	jobs_dir = Path(args.jobs_dir)
	try:
		asyncio.run(_worker_loop(jobs_dir, float(args.poll_s)))
		return 0
	except KeyboardInterrupt:
		return 0
	except Exception as e:
		print(f"[jump_clip_worker] fatal: {e!r}")
		return 2


if __name__ == "__main__":
	raise SystemExit(main())


