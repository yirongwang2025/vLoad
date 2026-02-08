from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from modules import db
from modules.config import get_config
from modules.video_backend import get_video_backend_for_tools
from modules.video_tools import cut_h264_clip_to_mp4_best_effort, extract_mp4_frame_times, probe_mp4_stream_info


def _safe_read_json(path: Path) -> Dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _read_session_t_start(session_dir: Path) -> Optional[float]:
	"""
	Read session start time (epoch seconds) from session.json.
	This is used to convert host-time windows into video-time offsets without needing frames.csv.
	"""
	try:
		p = session_dir / "session.json"
		if not p.exists():
			return None
		data = json.loads(p.read_text(encoding="utf-8"))
		t0 = data.get("t_start")
		return float(t0) if isinstance(t0, (int, float, str)) else None
	except Exception:
		return None


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
	event_id = int(job.get("event_id") or 0)
	session_id = str(job.get("session_id") or "").strip()
	if jump_id <= 0 or not session_id:
		return "invalid job (missing jump_id/session_id)"

	clip_start_host = float(job.get("clip_start_host"))
	clip_end_host = float(job.get("clip_end_host"))
	video_fps = int(job.get("video_fps") or 30)

	CFG = get_config()
	sessions_base = Path(CFG.sessions.base_dir or str(Path("data") / "sessions"))
	jump_clips_subdir = (CFG.sessions.jump_clips_subdir or "jump_clips").strip() or "jump_clips"

	base = sessions_base / session_id
	mp4 = base / "video.mp4"
	h264 = base / "video.h264"

	# If the session directory doesn't exist at all, this is a real failure (bad session_id / wrong base_dir).
	if not base.exists():
		return f"session dir not found: {base}"

	# Prefer MP4 if available (fast stream-copy cutting).
	# If MP4 isn't available yet (session still running), attempt a "real-time" clip from the growing H264.
	use_mp4 = bool(mp4.exists() and mp4.stat().st_size > 0)
	use_h264 = bool(h264.exists() and h264.stat().st_size > 0)
	if not use_mp4 and not use_h264:
		return "mp4/h264 not ready yet — retry later"

	# Compute clip offsets using host time directly, anchored by session.json t_start.
	# This removes any dependency on frames.csv or session-level frames in DB.
	t_start = _read_session_t_start(base)
	if t_start is None:
		return "session.json missing t_start (cannot cut clip by host time)"
	start_sec = max(0.0, clip_start_host - float(t_start))
	duration = max(0.2, clip_end_host - clip_start_host)

	clip_backend = get_video_backend_for_tools()

	clips_dir = base / jump_clips_subdir
	clips_dir.mkdir(parents=True, exist_ok=True)
	out_name = f"jump_{event_id}.mp4" if event_id > 0 else f"jump_{jump_id}.mp4"
	out_path = clips_dir / out_name

	# Cut clip:
	# - If MP4 exists: fast stream-copy cut
	# - Else: best-effort cut from raw H264 (may re-encode), enabling "real-time" clips
	ok_cut = False
	if use_mp4:
		try:
			ok_cut = bool(clip_backend.cut_clip_best_effort(mp4, out_path, start_sec=start_sec, duration_sec=duration))
		except Exception:
			ok_cut = False
	else:
		try:
			ok_cut = bool(cut_h264_clip_to_mp4_best_effort(h264, out_path, start_sec=start_sec, duration_sec=duration, fps=video_fps))
		except Exception:
			ok_cut = False

	if not ok_cut:
		return "clip cut failed — retry later"

	# Validate output is a playable MP4 before we publish it to the UI (set video_path).
	# Cutting from a growing H264 can occasionally produce an invalid/truncated MP4.
	try:
		if not (out_path.exists() and out_path.stat().st_size > 2048):
			try:
				if out_path.exists():
					out_path.unlink()
			except Exception:
				pass
			return "clip output too small/absent — retry later"
	except Exception:
		return "clip output stat failed — retry later"

	info = probe_mp4_stream_info(out_path)
	times = extract_mp4_frame_times(out_path)
	if not info or not times:
		# Treat as invalid; delete and retry later.
		try:
			out_path.unlink()
		except Exception:
			pass
		return "clip mp4 not valid/playable yet — retry later"

	# Resolve the canonical jump row id before persisting DB references (jobs can be stale).
	resolved_jump_id = await db.resolve_jump_row_id(jump_id=jump_id, event_id=event_id if event_id > 0 else None, session_id=session_id)
	if not resolved_jump_id:
		# Keep job pending; likely racing DB insert or referencing a cleared DB.
		return "jump row not found in DB yet — retry later"

	# Update jump video_path using resolved jump id (preferred)
	rel = str(sessions_base / session_id / jump_clips_subdir / out_name)
	await db.set_jump_video_path_by_jump_id(int(resolved_jump_id), rel)

	# Persist per-jump clip-relative frames (jump_frames) for sync.
	# Derive from the clip MP4 itself (ffprobe) => no frames.csv dependency.
	w = info.get("width")
	h = info.get("height")
	jf = []
	for i, tv in enumerate(times):
		try:
			tv_f = float(tv)
		except Exception:
			continue
		jf.append(
			{
				"frame_idx": int(i),
				"t_video": max(0.0, tv_f),
				"t_host": float(clip_start_host) + max(0.0, tv_f),
				"device_ts": None,
				"width": int(w) if isinstance(w, (int, float)) else None,
				"height": int(h) if isinstance(h, (int, float)) else None,
			}
		)
	await db.replace_jump_frames(int(resolved_jump_id), jf)

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


