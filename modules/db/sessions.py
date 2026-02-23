"""Sessions and frames: upsert_session_start, update_session_stop, replace_frames, get_frames."""
import logging
from typing import Any, Dict, List, Optional, Sequence

from modules.config import get_config
from modules.db.pool import get_pool, _to_dt

logger = logging.getLogger(__name__)


async def upsert_session_start(
	session_id: str,
	t_start: float,
	imu_mode: Optional[str],
	imu_rate: Optional[int],
	jump_config: Optional[Dict[str, Any]],
	video_backend: Optional[str] = None,
	video_fps: Optional[int] = None,
	video_path: Optional[str] = None,
	camera_clock_offset_s: Optional[float] = None,
	meta: Optional[Dict[str, Any]] = None,
) -> None:
	"""Create/update a session row at session start."""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] upsert_session_start: pool is None, skipping")
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			INSERT INTO sessions (session_id, t_start, imu_mode, imu_rate, video_backend, video_fps, video_path, camera_clock_offset_s, jump_config, meta)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
			ON CONFLICT (session_id) DO UPDATE SET
				t_start = EXCLUDED.t_start,
				imu_mode = EXCLUDED.imu_mode,
				imu_rate = EXCLUDED.imu_rate,
				video_backend = COALESCE(EXCLUDED.video_backend, sessions.video_backend),
				video_fps = EXCLUDED.video_fps,
				video_path = COALESCE(EXCLUDED.video_path, sessions.video_path),
				camera_clock_offset_s = COALESCE(EXCLUDED.camera_clock_offset_s, sessions.camera_clock_offset_s),
				jump_config = EXCLUDED.jump_config,
				meta = EXCLUDED.meta;
			""",
			sid,
			_to_dt(float(t_start)),
			imu_mode,
			int(imu_rate) if imu_rate is not None else None,
			video_backend,
			int(video_fps) if video_fps is not None else None,
			video_path,
			float(camera_clock_offset_s) if camera_clock_offset_s is not None else None,
			jump_config,
			meta,
		)


async def update_session_stop(session_id: str, t_stop: float, video_path: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
	"""Update t_stop and optionally video_path/meta for a session."""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] update_session_stop: pool is None, skipping")
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE sessions
			SET t_stop = $2, video_path = COALESCE($3, video_path), meta = COALESCE($4, meta)
			WHERE session_id = $1;
			""",
			sid,
			_to_dt(float(t_stop)),
			video_path,
			meta,
		)


async def update_session_camera_calibration(session_id: str, camera_clock_offset_s: Optional[float], video_backend: Optional[str] = None) -> None:
	"""Store timebase mapping hint for camera backends."""
	pool = get_pool()
	if pool is None:
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE sessions
			SET camera_clock_offset_s = COALESCE($2, camera_clock_offset_s),
			    video_backend = COALESCE($3, video_backend)
			WHERE session_id = $1;
			""",
			sid,
			(float(camera_clock_offset_s) if camera_clock_offset_s is not None else None),
			video_backend,
		)


async def replace_frames(session_id: str, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""Replace all frames for a session (delete then bulk insert)."""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] replace_frames: pool is None, skipping")
		return {"inserted": 0}
	sid = (session_id or "").strip()
	if not sid:
		return {"inserted": 0}
	async with pool.acquire() as conn:
		async with conn.transaction():
			await conn.execute("DELETE FROM frames WHERE session_id = $1;", sid)
			if not frames:
				return {"inserted": 0}
			records = []
			for f in frames:
				try:
					records.append(
						(
							sid,
							int(f.get("frame_idx")),
							float(f.get("t_host")),
							(float(f.get("device_ts")) if f.get("device_ts") is not None and f.get("device_ts") != "" else None),
							(str(f.get("source")) if f.get("source") not in (None, "") else None),
							(int(f.get("width")) if f.get("width") is not None and f.get("width") != "" else None),
							(int(f.get("height")) if f.get("height") is not None and f.get("height") != "" else None),
						)
					)
				except Exception:
					continue
			if not records:
				return {"inserted": 0}
			await conn.copy_records_to_table(
				"frames",
				records=records,
				columns=["session_id", "frame_idx", "t_host", "device_ts", "source", "width", "height"],
			)
			return {"inserted": len(records)}


async def get_frames(
	session_id: str,
	limit: Optional[int] = None,
	t0: Optional[float] = None,
	t1: Optional[float] = None,
) -> List[Dict[str, Any]]:
	"""Return frames for a session from DB. Optional host-time filtering."""
	pool = get_pool()
	if pool is None:
		return []
	sid = (session_id or "").strip()
	if not sid:
		return []
	if limit is None:
		limit = int(get_config().api.session_frames_limit_default)
	lim = max(1, min(int(limit), int(get_config().runtime.backfill_frames_limit)))
	async with pool.acquire() as conn:
		if t0 is not None and t1 is not None:
			rows = await conn.fetch(
				"""
				SELECT frame_idx, t_host, device_ts, source, width, height
				FROM frames WHERE session_id = $1 AND t_host >= $2 AND t_host <= $3
				ORDER BY frame_idx ASC LIMIT $4;
				""",
				sid, float(t0), float(t1), lim,
			)
		else:
			rows = await conn.fetch(
				"""
				SELECT frame_idx, t_host, device_ts, source, width, height
				FROM frames WHERE session_id = $1 ORDER BY frame_idx ASC LIMIT $2;
				""",
				sid, lim,
			)
	out: List[Dict[str, Any]] = []
	for r in rows:
		out.append({
			"frame_idx": int(r["frame_idx"]),
			"t_host": float(r["t_host"]),
			"device_ts": float(r["device_ts"]) if r["device_ts"] is not None else None,
			"source": str(r["source"]) if r.get("source") is not None else None,
			"width": int(r["width"]) if r["width"] is not None else None,
			"height": int(r["height"]) if r["height"] is not None else None,
		})
	return out
