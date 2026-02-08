"""Session and debug routes. Routes: /session/start, /session/stop, /session/status; /sessions/{id}/video, /sessions/{id}/frames; /debug/status."""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from modules import db
from schemas.requests import SessionStartPayload

import state

router = APIRouter(tags=["sessions"])


@router.post("/session/start")
async def session_start(payload: SessionStartPayload):
	"""Start a recording session. Creates session directory, video recording, frames.csv, IMU CSV."""
	async with state._session_lock:
		if state._session_id is not None:
			return {"detail": "Session already running", "session_id": state._session_id}
		sid = (payload.session_id or "").strip()
		if not sid:
			sid = time.strftime("%Y%m%d_%H%M%S")
		base = state._session_base_dir(sid)
		base.mkdir(parents=True, exist_ok=True)
		t_start = time.time()
		try:
			(base / "session.json").write_text(
				json.dumps(
					{
						"session_id": sid,
						"t_start": t_start,
						"imu": {"mode": state._active_mode, "rate": state._active_rate},
						"jump_config": state._jump_config,
					},
					indent=2,
				),
				encoding="utf-8",
			)
		except Exception:
			pass
		try:
			base_root = Path(state.CFG.sessions.base_dir or str(Path("data") / "sessions"))
			await db.upsert_session_start(
				session_id=sid,
				t_start=t_start,
				imu_mode=state._active_mode,
				imu_rate=state._active_rate,
				jump_config=state._jump_config,
				video_backend=(state._video.name() or "video").strip(),
				video_fps=30,
				video_path=str(base_root / sid / "video.mp4"),
				camera_clock_offset_s=None,
				meta=None,
			)
		except Exception:
			pass
		try:
			state._video.start()
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Video backend start failed: {e!r}")
		try:
			deadline = time.time() + 3.0
			last_status = None
			while time.time() < deadline:
				st = state._video.get_status()
				last_status = st
				if st.get("error"):
					raise HTTPException(status_code=500, detail=f"Video backend error after start(): {st.get('error')}")
				if bool(st.get("running")):
					break
				await asyncio.sleep(0.1)
			else:
				raise HTTPException(status_code=500, detail=f"Video backend did not become running: {last_status!r}")
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Video backend readiness check failed: {e!r}")
		try:
			state._video.start_recording(str(base), fps=30)
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Video backend start_recording raised: {e!r}")
		try:
			await asyncio.sleep(0.05)
			st = state._video.get_status()
			if not bool(st.get("recording")):
				raise HTTPException(status_code=500, detail=f"Video recording did not start: {st.get('error') or st!r}")
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Video recording status check failed: {e!r}")
		imu_path = base / "imu.csv"
		imu_new = not imu_path.exists()
		state._imu_csv_fh = open(imu_path, "a", encoding="utf-8", newline="\n")
		if imu_new:
			state._imu_csv_fh.write("t,imu_timestamp,imu_sample_index,ax,ay,az,gx,gy,gz,mx,my,mz\n")
			state._imu_csv_fh.flush()
		state._session_id = sid
		state._session_dir = base
		return {"detail": "Session started", "session_id": sid, "dir": str(base)}


@router.post("/session/stop")
async def session_stop():
	"""Stop the active recording session."""
	async with state._session_lock:
		if state._session_id is None:
			return {"detail": "No active session"}
		sid = state._session_id
		try:
			state._video.stop_recording()
		except Exception:
			pass
		try:
			if state._session_dir is not None:
				state._video.mux_to_mp4_best_effort(state._session_dir, fps=30)
				try:
					h264 = state._session_dir / "video.h264"
					mp4 = state._session_dir / "video.mp4"
					if mp4.exists() and mp4.stat().st_size > 0 and h264.exists():
						h264.unlink()
				except Exception:
					pass
		except Exception:
			pass
		try:
			if state._imu_csv_fh:
				state._imu_csv_fh.close()
		except Exception:
			pass
		state._imu_csv_fh = None
		t_stop = time.time()
		try:
			if state._session_dir:
				path = state._session_dir / "session.json"
				if path.exists():
					data = json.loads(path.read_text(encoding="utf-8"))
				else:
					data = {}
				data["t_stop"] = t_stop
				path.write_text(json.dumps(data, indent=2), encoding="utf-8")
		except Exception:
			pass
		try:
			base_root = Path(state.CFG.sessions.base_dir or str(Path("data") / "sessions"))
			await db.update_session_stop(
				session_id=sid,
				t_stop=t_stop,
				video_path=str(base_root / sid / "video.mp4"),
				meta=None,
			)
		except Exception:
			pass
		try:
			if state._session_dir:
				p = state._session_dir / "frames.csv"
				if p.exists():
					video_backend = (state._video.name() or "video").strip()
					out = []
					with open(p, "r", encoding="utf-8") as fh:
						_ = fh.readline()
						for line in fh:
							parts = line.strip().split(",")
							if len(parts) < 5:
								continue
							try:
								out.append({
									"frame_idx": int(parts[0]),
									"t_host": float(parts[1]),
									"device_ts": float(parts[2]) if parts[2] else None,
									"source": video_backend,
									"width": int(parts[3]) if parts[3] else None,
									"height": int(parts[4]) if parts[4] else None,
								})
							except Exception:
								continue
					await db.replace_frames(sid, out)
					try:
						first = next((f for f in out if f.get("device_ts") is not None), None)
						if first and first.get("device_ts") is not None:
							offset = float(first["t_host"]) - float(first["device_ts"])
							await db.update_session_camera_calibration(sid, camera_clock_offset_s=offset, video_backend=video_backend)
					except Exception:
						pass
		except Exception:
			pass
		state._session_id = None
		state._session_dir = None
		return {"detail": "Session stopped", "session_id": sid}


@router.get("/session/status")
async def session_status():
	return {"session_id": state._session_id, "dir": str(state._session_dir) if state._session_dir else None}


@router.get("/sessions/{session_id}/video")
async def get_session_video(session_id: str):
	"""Serve a session video file for playback. Prefers video.mp4; may mux from video.h264."""
	base = state._session_base_dir(session_id)
	mp4 = base / "video.mp4"
	if mp4.exists():
		return FileResponse(str(mp4), media_type="video/mp4", filename="video.mp4")
	try:
		h264 = base / "video.h264"
		if h264.exists():
			try:
				state._video.mux_to_mp4_best_effort(base, fps=30)
			except Exception:
				pass
			deadline = time.time() + 15.0
			while time.time() < deadline:
				if mp4.exists() and mp4.stat().st_size > 0:
					try:
						if h264.exists():
							h264.unlink()
					except Exception:
						pass
					return FileResponse(str(mp4), media_type="video/mp4", filename="video.mp4")
				await asyncio.sleep(0.25)
	except Exception:
		pass
	raise HTTPException(
		status_code=404,
		detail=f"video.mp4 not found for session {session_id!r}. Stop the session to trigger mux, or convert from video.h264.",
	)


@router.get("/sessions/{session_id}/frames")
async def get_session_frames(session_id: str, t0: float = None, t1: float = None, limit: int = 200000):
	"""Return per-frame timing as JSON. Prefers in-memory buffer, then DB, then frames.csv."""
	if session_id == state._session_id and state._frame_history:
		try:
			frames = list(state._frame_history)
			if t0 is not None or t1 is not None:
				filtered = []
				for f in frames:
					th = f.get("t_host")
					if not isinstance(th, (int, float)):
						continue
					th_f = float(th)
					if t0 is not None and th_f < float(t0):
						continue
					if t1 is not None and th_f > float(t1):
						continue
					filtered.append(f)
				frames = filtered
			if limit > 0 and len(frames) > limit:
				frames = frames[:limit]
			if frames:
				video_backend = (state._video.name() or "video").strip()
				for f in frames:
					if "source" not in f:
						f["source"] = video_backend
				return {"count": len(frames), "frames": frames, "source": "memory"}
		except Exception:
			pass
	try:
		frames = await db.get_frames(session_id=session_id, limit=limit, t0=t0, t1=t1)
		if frames:
			return {"count": len(frames), "frames": frames, "source": "db"}
	except Exception:
		pass
	base = state._session_base_dir(session_id)
	p = base / "frames.csv"
	if not p.exists():
		raise HTTPException(status_code=404, detail="frames.csv not found")
	out = []
	try:
		with open(p, "r", encoding="utf-8") as fh:
			fh.readline()
			for line in fh:
				parts = line.strip().split(",")
				if len(parts) < 5:
					continue
				try:
					video_backend = (state._video.name() or "video").strip()
					frame_data = {
						"frame_idx": int(parts[0]),
						"t_host": float(parts[1]),
						"device_ts": float(parts[2]) if parts[2] else None,
						"source": video_backend,
						"width": int(parts[3]),
						"height": int(parts[4]),
					}
					if t0 is not None and frame_data["t_host"] < float(t0):
						continue
					if t1 is not None and frame_data["t_host"] > float(t1):
						continue
					out.append(frame_data)
				except Exception:
					continue
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to read frames.csv: {e!r}")
	return {"count": len(out), "frames": out, "source": "file"}


@router.get("/debug/status")
async def debug_status():
	"""Quick diagnostics: IMU flow, detection enabled, jump worker, queue depth."""
	q_size = None
	q_max = None
	try:
		if state._jump_sample_queue is not None:
			q_size = state._jump_sample_queue.qsize()
			q_max = getattr(state._jump_sample_queue, "_maxsize", None)
	except Exception:
		pass
	try:
		if state._count_clip_jobs_pending:
			state._dbg["clip_jobs_pending"] = int(state._count_clip_jobs_pending())
	except Exception:
		pass
	try:
		if state._count_clip_jobs_done:
			state._dbg["clip_jobs_done"] = int(state._count_clip_jobs_done())
	except Exception:
		pass
	try:
		if state._count_clip_jobs_failed:
			state._dbg["clip_jobs_failed"] = int(state._count_clip_jobs_failed())
	except Exception:
		pass
	try:
		if state._read_last_clip_job_error:
			state._dbg["clip_last_failed"] = state._read_last_clip_job_error()
	except Exception:
		pass
	return {
		"detection_enabled": bool(state._jump_detection_enabled),
		"active_mode": state._active_mode,
		"active_rate": state._active_rate,
		"jump_config": state._jump_config,
		"session_id": state._session_id,
		"queue_size": q_size,
		"queue_maxsize": q_max,
		"dbg": dict(state._dbg),
	}
