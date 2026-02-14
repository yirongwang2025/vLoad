"""Jumps, jump_frames, imu_samples, annotations."""
import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

from modules.db.helpers import frame_row_to_dict, imu_sample_row_to_dict, jump_row_to_dict
from modules.db.pool import get_pool, _to_dt

logger = logging.getLogger(__name__)

async def replace_jump_frames(jump_id: int, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Replace all frames for a jump (delete then bulk insert).
	Frames are expected to be clip-relative: frame_idx starts at 0, t_video starts at 0.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] replace_jump_frames: pool is None, skipping")
		return {"inserted": 0}
	jid = int(jump_id)
	async with pool.acquire() as conn:
		async with conn.transaction():
			await conn.execute("DELETE FROM jump_frames WHERE jump_id = $1;", jid)
			if not frames:
				return {"inserted": 0}
			records = []
			for f in frames:
				try:
					records.append(
						(
							jid,
							int(f.get("frame_idx")),
							float(f.get("t_video")),
							float(f.get("t_host")),
							(float(f.get("device_ts")) if f.get("device_ts") is not None and f.get("device_ts") != "" else None),
							(int(f.get("width")) if f.get("width") is not None and f.get("width") != "" else None),
							(int(f.get("height")) if f.get("height") is not None and f.get("height") != "" else None),
						)
					)
				except Exception:
					continue
			if not records:
				return {"inserted": 0}
			await conn.copy_records_to_table(
				"jump_frames",
				records=records,
				columns=["jump_id", "frame_idx", "t_video", "t_host", "device_ts", "width", "height"],
			)
			return {"inserted": len(records)}


async def get_jump_frames(jump_id: int, limit: int = 200000) -> List[Dict[str, Any]]:
	"""
	Get clip-relative frames for a jump.
	"""
	pool = get_pool()
	if pool is None:
		return []
	jid = int(jump_id)
	lim = max(1, min(int(limit), 500000))
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT frame_idx, t_video, t_host, device_ts, width, height
			FROM jump_frames
			WHERE jump_id = $1
			ORDER BY frame_idx ASC
			LIMIT $2;
			""",
			jid,
			lim,
		)
	return [frame_row_to_dict(r) for r in rows]


async def insert_jump_with_imu(
	jump: Dict[str, Any],
	annotation: Dict[str, Any],
	imu_samples: Sequence[Dict[str, Any]],
) -> Optional[int]:
	"""
	Insert one jump row and its associated IMU samples into the database.

	- jump: dict containing event_id, t_peak, t_start, t_end, and metrics.
	- annotation: dict with optional 'name', 'note'.
	- imu_samples: list of rows from _imu_history (t, imu_timestamp, acc, gyro, mag).
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] insert_jump_with_imu: pool is None, skipping")
		return None
	logger.debug("[DB] insert_jump_with_imu: session_id=%s, event_id=%s, samples=%s", jump.get("session_id"), jump.get("event_id"), len(imu_samples))

	event_id = int(jump.get("event_id", 0)) or None
	session_id = jump.get("session_id")
	video_path = jump.get("video_path")
	t_peak_raw = jump.get("t_peak", None)
	if t_peak_raw is None:
		raise ValueError("jump.t_peak is None (DB requires NOT NULL)")
	try:
		t_peak = float(t_peak_raw)
	except (TypeError, ValueError) as e:
		raise ValueError(f"jump.t_peak is not a float: {t_peak_raw!r}") from e
	t_start = jump.get("t_start")
	t_end = jump.get("t_end")

	t_peak_dt = _to_dt(t_peak)
	t_start_dt = _to_dt(float(t_start)) if t_start is not None else None
	t_end_dt = _to_dt(float(t_end)) if t_end is not None else None
	t_takeoff_calc = jump.get("t_takeoff")
	t_landing_calc = jump.get("t_landing")
	t_takeoff_calc_dt = _to_dt(float(t_takeoff_calc)) if t_takeoff_calc is not None else None
	t_landing_calc_dt = _to_dt(float(t_landing_calc)) if t_landing_calc is not None else None

	# Original algorithm rotation metrics (optional)
	theta_z_rad = jump.get("theta_z_rad")
	revolutions_est = jump.get("revolutions_est")
	revolutions_class = jump.get("revolutions_class")
	underrotation = jump.get("underrotation")
	underrot_flag = jump.get("underrot_flag")
	gz_bias = jump.get("gz_bias")

	async with pool.acquire() as conn:
		# Insert into jumps table (idempotent for a given (session_id, event_id)).
		# We avoid overwriting name/note if a user already saved them.
		if session_id is not None and event_id is not None:
			jump_id = await conn.fetchval(
				"""
				INSERT INTO jumps (
					event_id, session_id, video_path, t_peak, t_start, t_end,
					t_takeoff_calc, t_landing_calc,
					theta_z_rad, revolutions_est, revolutions_class, underrotation, underrot_flag, gz_bias,
					flight_time, height, acc_peak, gyro_peak,
					rotation_phase, confidence, name, note
				)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
				ON CONFLICT (session_id, event_id) DO UPDATE SET
					-- Keep the latest computed metrics, but do not clobber user annotations.
					video_path = COALESCE(EXCLUDED.video_path, jumps.video_path),
					t_peak = EXCLUDED.t_peak,
					t_start = EXCLUDED.t_start,
					t_end = EXCLUDED.t_end,
					t_takeoff_calc = COALESCE(EXCLUDED.t_takeoff_calc, jumps.t_takeoff_calc),
					t_landing_calc = COALESCE(EXCLUDED.t_landing_calc, jumps.t_landing_calc),
					theta_z_rad = COALESCE(EXCLUDED.theta_z_rad, jumps.theta_z_rad),
					revolutions_est = COALESCE(EXCLUDED.revolutions_est, jumps.revolutions_est),
					revolutions_class = COALESCE(EXCLUDED.revolutions_class, jumps.revolutions_class),
					underrotation = COALESCE(EXCLUDED.underrotation, jumps.underrotation),
					underrot_flag = COALESCE(EXCLUDED.underrot_flag, jumps.underrot_flag),
					gz_bias = COALESCE(EXCLUDED.gz_bias, jumps.gz_bias),
					flight_time = COALESCE(EXCLUDED.flight_time, jumps.flight_time),
					height = COALESCE(EXCLUDED.height, jumps.height),
					acc_peak = COALESCE(EXCLUDED.acc_peak, jumps.acc_peak),
					gyro_peak = COALESCE(EXCLUDED.gyro_peak, jumps.gyro_peak),
					rotation_phase = COALESCE(EXCLUDED.rotation_phase, jumps.rotation_phase),
					confidence = COALESCE(EXCLUDED.confidence, jumps.confidence),
					name = COALESCE(jumps.name, EXCLUDED.name),
					note = COALESCE(jumps.note, EXCLUDED.note)
				RETURNING id;
				""",
				event_id,
				session_id,
				video_path,
				t_peak_dt,
				t_start_dt,
				t_end_dt,
				t_takeoff_calc_dt,
				t_landing_calc_dt,
				(float(theta_z_rad) if theta_z_rad is not None else None),
				(float(revolutions_est) if revolutions_est is not None else None),
				(int(revolutions_class) if revolutions_class is not None else None),
				(float(underrotation) if underrotation is not None else None),
				(bool(underrot_flag) if underrot_flag is not None else None),
				(float(gz_bias) if gz_bias is not None else None),
				jump.get("flight_time"),
				jump.get("height"),
				jump.get("acc_peak"),
				jump.get("gyro_peak"),
				jump.get("rotation_phase"),
				jump.get("confidence"),
				annotation.get("name"),
				annotation.get("note"),
			)
		else:
			# Fallback (should be rare): no stable idempotency key available.
			jump_id = await conn.fetchval(
				"""
				INSERT INTO jumps (
					event_id, session_id, video_path, t_peak, t_start, t_end,
					t_takeoff_calc, t_landing_calc,
					theta_z_rad, revolutions_est, revolutions_class, underrotation, underrot_flag, gz_bias,
					flight_time, height, acc_peak, gyro_peak,
					rotation_phase, confidence, name, note
				)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
				RETURNING id;
				""",
				event_id,
				session_id,
				video_path,
				t_peak_dt,
				t_start_dt,
				t_end_dt,
				t_takeoff_calc_dt,
				t_landing_calc_dt,
				(float(theta_z_rad) if theta_z_rad is not None else None),
				(float(revolutions_est) if revolutions_est is not None else None),
				(int(revolutions_class) if revolutions_class is not None else None),
				(float(underrotation) if underrotation is not None else None),
				(bool(underrot_flag) if underrot_flag is not None else None),
				(float(gz_bias) if gz_bias is not None else None),
				jump.get("flight_time"),
				jump.get("height"),
				jump.get("acc_peak"),
				jump.get("gyro_peak"),
				jump.get("rotation_phase"),
				jump.get("confidence"),
				annotation.get("name"),
				annotation.get("note"),
			)

		if not imu_samples:
			return int(jump_id) if jump_id is not None else None

		# Avoid duplicating IMU rows if this jump already has samples (idempotency).
		try:
			existing_cnt = await conn.fetchval("SELECT COUNT(*) FROM imu_samples WHERE jump_id = $1;", int(jump_id))
			existing_cnt_i = int(existing_cnt or 0)
		except Exception:
			existing_cnt_i = 0
		# If we already have at least as many samples as we are about to insert, assume it's done.
		if existing_cnt_i > 0 and existing_cnt_i >= len(imu_samples):
			return int(jump_id) if jump_id is not None else None
		# Otherwise, replace (handles rare cases where an earlier attempt inserted an incomplete window).
		if existing_cnt_i > 0:
			try:
				await conn.execute("DELETE FROM imu_samples WHERE jump_id = $1;", int(jump_id))
			except Exception:
				pass

		rows: List[List[Any]] = []
		for s in imu_samples:
			t = float(s.get("t", 0.0))
			imu_ts = s.get("imu_timestamp")
			acc = s.get("acc") or []
			gyro = s.get("gyro") or []
			mag = s.get("mag") or []
			acc_x = float(acc[0]) if len(acc) > 0 else None
			acc_y = float(acc[1]) if len(acc) > 1 else None
			acc_z = float(acc[2]) if len(acc) > 2 else None
			gyro_x = float(gyro[0]) if len(gyro) > 0 else None
			gyro_y = float(gyro[1]) if len(gyro) > 1 else None
			gyro_z = float(gyro[2]) if len(gyro) > 2 else None
			mag_x = float(mag[0]) if len(mag) > 0 else None
			mag_y = float(mag[1]) if len(mag) > 1 else None
			mag_z = float(mag[2]) if len(mag) > 2 else None
			rows.append(
				[
					jump_id,
					_to_dt(t),
					imu_ts,
					acc_x,
					acc_y,
					acc_z,
					gyro_x,
					gyro_y,
					gyro_z,
					mag_x,
					mag_y,
					mag_z,
				]
			)

		await conn.executemany(
			"""
			INSERT INTO imu_samples (
				jump_id, t, imu_timestamp,
				acc_x, acc_y, acc_z,
				gyro_x, gyro_y, gyro_z,
				mag_x, mag_y, mag_z
			)
			VALUES (
				$1,$2,$3,
				$4,$5,$6,
				$7,$8,$9,
				$10,$11,$12
			);
			""",
			rows,
		)
		return int(jump_id) if jump_id is not None else None


async def update_annotation(event_id: int, name: Optional[str], note: Optional[str]) -> None:
	"""
	Update the name/note annotation for a jump row identified by event_id.

	If the database is not configured or no such event_id exists, this is a no‑op.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] update_annotation: pool is None, skipping")
		return
	logger.debug("[DB] update_annotation: event_id=%s, name=%r", event_id, name)

	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET name = $1,
			    note = $2
			WHERE event_id = $3;
			""",
			name,
			note,
			event_id,
		)


async def update_annotation_by_jump_id(jump_id: int, name: Optional[str], note: Optional[str]) -> None:
	"""
	Update the name/note annotation for a jump row identified by jumps.id (jump_id).

	This is the preferred method because event_id is not guaranteed unique across sessions.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] update_annotation_by_jump_id: pool is None, skipping")
		return
	jid = int(jump_id)
	logger.debug("[DB] update_annotation_by_jump_id: jump_id=%s, name=%r", jid, name)

	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET name = $1,
			    note = $2
			WHERE id = $3;
			""",
			name,
			note,
			jid,
		)


async def update_jump_video_mark(event_id: int, which: str, t_host: Optional[float], t_video: Optional[float]) -> Dict[str, Any]:
	"""
	Store video-verified takeoff/landing marks for a jump identified by event_id.

	- which: "start"/"takeoff" or "end"/"landing"
	- t_host: host epoch seconds corresponding to the marked frame
	- t_video: clip-relative video seconds (video.currentTime)
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	w = (which or "").strip().lower()
	if w in ("start", "takeoff", "liftoff"):
		col_ts = "t_takeoff_video"
		col_tv = "t_takeoff_video_t"
	elif w in ("end", "landing", "touchdown"):
		col_ts = "t_landing_video"
		col_tv = "t_landing_video_t"
	else:
		return {"ok": False, "error": "which must be start/end (takeoff/landing)"}

	# Convert host time to TIMESTAMPTZ, allow clearing by sending nulls.
	t_ts = _to_dt(float(t_host)) if t_host is not None else None
	t_v = float(t_video) if t_video is not None else None

	async with pool.acquire() as conn:
		await conn.execute(
			f"""
			UPDATE jumps
			SET {col_ts} = $1,
			    {col_tv} = $2
			WHERE event_id = $3;
			""",
			t_ts,
			t_v,
			int(event_id),
		)
		# Return current values so UI can refresh without another fetch.
		row = await conn.fetchrow(
			"""
			SELECT id, event_id,
			       t_takeoff_video, t_takeoff_video_t,
			       t_landing_video, t_landing_video_t
			FROM jumps
			WHERE event_id = $1
			ORDER BY created_at DESC
			LIMIT 1;
			""",
			int(event_id),
		)
		if not row:
			return {"ok": False, "error": "jump not found"}
		# If both marks exist, recompute IMU-marked metrics (best-effort).
		try:
			if row["t_takeoff_video"] is not None and row["t_landing_video"] is not None:
				await recompute_marked_imu_metrics(event_id=int(event_id))
		except Exception:
			pass
		return {
			"ok": True,
			"event_id": row["event_id"],
			"t_takeoff_video": row["t_takeoff_video"].timestamp() if row["t_takeoff_video"] else None,
			"t_takeoff_video_t": float(row["t_takeoff_video_t"]) if row["t_takeoff_video_t"] is not None else None,
			"t_landing_video": row["t_landing_video"].timestamp() if row["t_landing_video"] else None,
			"t_landing_video_t": float(row["t_landing_video_t"]) if row["t_landing_video_t"] is not None else None,
		}


async def update_jump_video_mark_by_jump_id(jump_id: int, which: str, t_host: Optional[float], t_video: Optional[float]) -> Dict[str, Any]:
	"""
	Store video-verified takeoff/landing marks for a jump identified by jumps.id (jump_id).

	This is the preferred method because event_id is not guaranteed unique across sessions.
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	jid = int(jump_id)
	w = (which or "").strip().lower()
	if w in ("start", "takeoff", "liftoff"):
		col_ts = "t_takeoff_video"
		col_tv = "t_takeoff_video_t"
	elif w in ("end", "landing", "touchdown"):
		col_ts = "t_landing_video"
		col_tv = "t_landing_video_t"
	else:
		return {"ok": False, "error": "which must be start/end (takeoff/landing)"}

	# Convert host time to TIMESTAMPTZ, allow clearing by sending nulls.
	t_ts = _to_dt(float(t_host)) if t_host is not None else None
	t_v = float(t_video) if t_video is not None else None

	async with pool.acquire() as conn:
		await conn.execute(
			f"""
			UPDATE jumps
			SET {col_ts} = $1,
			    {col_tv} = $2
			WHERE id = $3;
			""",
			t_ts,
			t_v,
			jid,
		)
		row = await conn.fetchrow(
			"""
			SELECT id, event_id,
			       t_takeoff_video, t_takeoff_video_t,
			       t_landing_video, t_landing_video_t
			FROM jumps
			WHERE id = $1
			LIMIT 1;
			""",
			jid,
		)
		if not row:
			return {"ok": False, "error": "jump not found"}
		# If both marks exist, recompute IMU-marked metrics (best-effort).
		try:
			if row["t_takeoff_video"] is not None and row["t_landing_video"] is not None:
				await recompute_marked_imu_metrics_by_jump_id(jump_id=jid)
		except Exception:
			pass
		return {
			"ok": True,
			"jump_id": int(row["id"]),
			"event_id": row["event_id"],
			"t_takeoff_video": row["t_takeoff_video"].timestamp() if row["t_takeoff_video"] else None,
			"t_takeoff_video_t": float(row["t_takeoff_video_t"]) if row["t_takeoff_video_t"] is not None else None,
			"t_landing_video": row["t_landing_video"].timestamp() if row["t_landing_video"] else None,
			"t_landing_video_t": float(row["t_landing_video_t"]) if row["t_landing_video_t"] is not None else None,
		}


async def recompute_marked_imu_metrics(event_id: int) -> Dict[str, Any]:
	"""
	Compute IMU-based rotation/height/flight_time using video-verified takeoff/landing marks.
	Persists results into jumps.*_marked columns.
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	ev = int(event_id)
	g_m_s2 = 9.80665

	async with pool.acquire() as conn:
		j = await conn.fetchrow(
			"""
			SELECT id, event_id, t_takeoff_video, t_landing_video
			FROM jumps
			WHERE event_id = $1
			ORDER BY created_at DESC
			LIMIT 1;
			""",
			ev,
		)
		if not j:
			return {"ok": False, "error": "jump not found"}
		jump_id = int(j["id"])
		t0_dt = j["t_takeoff_video"]
		t1_dt = j["t_landing_video"]
		if t0_dt is None or t1_dt is None:
			return {"ok": False, "error": "marks not set"}
		t0 = float(t0_dt.timestamp())
		t1 = float(t1_dt.timestamp())
		if not (t1 > t0):
			return {"ok": False, "error": "invalid marks (end must be after start)"}

		# Gyro bias window: [t0-0.5, t0-0.1]
		b0 = t0 - 0.5
		b1 = t0 - 0.1
		bias_rows = await conn.fetch(
			"""
			SELECT EXTRACT(EPOCH FROM t) AS t_s, gyro_x, gyro_y, gyro_z
			FROM imu_samples
			WHERE jump_id = $1 AND t >= to_timestamp($2) AND t <= to_timestamp($3)
			ORDER BY t ASC;
			""",
			jump_id,
			float(b0),
			float(b1),
		)
		bias_x: list[float] = []
		bias_y: list[float] = []
		bias_z: list[float] = []
		for r in bias_rows:
			try:
				if r["gyro_x"] is not None:
					bias_x.append(float(r["gyro_x"]))
				if r["gyro_y"] is not None:
					bias_y.append(float(r["gyro_y"]))
				if r["gyro_z"] is not None:
					bias_z.append(float(r["gyro_z"]))
			except Exception:
				continue
		gx_bias = float(statistics.median(bias_x)) if bias_x else 0.0
		gy_bias = float(statistics.median(bias_y)) if bias_y else 0.0
		gz_bias = float(statistics.median(bias_z)) if bias_z else 0.0

		rows = await conn.fetch(
			"""
			SELECT EXTRACT(EPOCH FROM t) AS t_s, gyro_x, gyro_y, gyro_z
			FROM imu_samples
			WHERE jump_id = $1 AND t >= to_timestamp($2) AND t <= to_timestamp($3)
			ORDER BY t ASC;
			""",
			jump_id,
			float(t0),
			float(t1),
		)
		ts: list[float] = []
		gx: list[float] = []
		gy: list[float] = []
		gz: list[float] = []
		for r in rows:
			try:
				t_s = float(r["t_s"])
			except Exception:
				continue
			try:
				vx = float(r["gyro_x"]) - gx_bias if r["gyro_x"] is not None else 0.0
				vy = float(r["gyro_y"]) - gy_bias if r["gyro_y"] is not None else 0.0
				vz = float(r["gyro_z"]) - gz_bias if r["gyro_z"] is not None else 0.0
			except Exception:
				continue
			ts.append(t_s)
			gx.append(vx)
			gy.append(vy)
			gz.append(vz)

		# Compute angular speed magnitude (more robust than relying on sensor Z axis).
		omega: list[float] = []
		for i in range(len(ts)):
			try:
				omega.append(math.sqrt(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]))
			except Exception:
				omega.append(0.0)

		# Heuristic unit detection:
		# - If gyro magnitude values are small (e.g., ~5-25), they are likely rad/s.
		# - If they are large (e.g., hundreds+), they are likely deg/s.
		max_abs_omega = max((abs(v) for v in omega), default=0.0)
		assume_rad_s = bool(max_abs_omega < 50.0)
		deg_to_rad = math.pi / 180.0

		theta_rad = 0.0
		t_peak = None
		if len(ts) >= 2:
			for i in range(1, len(ts)):
				dt = ts[i] - ts[i - 1]
				if dt <= 0:
					continue
				w0 = omega[i - 1]
				w1 = omega[i]
				if not assume_rad_s:
					w0 *= deg_to_rad
					w1 *= deg_to_rad
				theta_rad += 0.5 * (w0 + w1) * dt
			# Peak time for phase uses omega magnitude
			try:
				i_peak = max(range(len(omega)), key=lambda k: abs(omega[k]))
				t_peak = ts[i_peak]
			except Exception:
				t_peak = None

		revolutions_est = abs(theta_rad) / (2.0 * math.pi) if math.pi else 0.0
		revolutions_class = int(round(revolutions_est))
		underrotation = float(revolutions_class) - float(revolutions_est)
		underrot_flag = bool(underrotation < -0.25)
		flight_time = float(t1 - t0)
		height = g_m_s2 * (flight_time ** 2) / 8.0
		phase = None
		if t_peak is not None and flight_time > 1e-6:
			phase = float((t_peak - t0) / flight_time)

		await conn.execute(
			"""
			UPDATE jumps
			SET flight_time_marked = $1,
			    height_marked = $2,
			    rotation_phase_marked = $3,
			    theta_z_rad_marked = $4,
			    revolutions_est_marked = $5,
			    revolutions_class_marked = $6,
			    underrotation_marked = $7,
			    underrot_flag_marked = $8,
			    gz_bias_marked = $9
			WHERE id = $10;
			""",
			float(flight_time),
			float(height),
			(float(phase) if phase is not None else None),
			float(theta_rad),
			float(revolutions_est),
			int(revolutions_class),
			float(underrotation),
			bool(underrot_flag),
			float(gz_bias),
			jump_id,
		)

		return {
			"ok": True,
			"event_id": ev,
			"flight_time_marked": float(flight_time),
			"height_marked": float(height),
			"theta_z_rad_marked": float(theta_rad),
			"revolutions_est_marked": float(revolutions_est),
			"revolutions_class_marked": int(revolutions_class),
			"underrotation_marked": float(underrotation),
			"underrot_flag_marked": bool(underrot_flag),
			"gz_bias_marked": float(gz_bias),
			"rotation_phase_marked": float(phase) if phase is not None else None,
		}


async def recompute_marked_imu_metrics_by_jump_id(jump_id: int) -> Dict[str, Any]:
	"""
	Same as recompute_marked_imu_metrics(), but targets a specific jump row by jumps.id (jump_id).
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	jid = int(jump_id)
	g_m_s2 = 9.80665

	async with pool.acquire() as conn:
		j = await conn.fetchrow(
			"""
			SELECT id, event_id, t_takeoff_video, t_landing_video
			FROM jumps
			WHERE id = $1
			LIMIT 1;
			""",
			jid,
		)
		if not j:
			return {"ok": False, "error": "jump not found"}
		ev = int(j["event_id"]) if j.get("event_id") is not None else None
		t0_dt = j["t_takeoff_video"]
		t1_dt = j["t_landing_video"]
		if t0_dt is None or t1_dt is None:
			return {"ok": False, "error": "marks not set"}
		t0 = float(t0_dt.timestamp())
		t1 = float(t1_dt.timestamp())
		if not (t1 > t0):
			return {"ok": False, "error": "invalid marks (end must be after start)"}

		# Gyro bias window: [t0-0.5, t0-0.1]
		b0 = t0 - 0.5
		b1 = t0 - 0.1
		bias_rows = await conn.fetch(
			"""
			SELECT EXTRACT(EPOCH FROM t) AS t_s, gyro_x, gyro_y, gyro_z
			FROM imu_samples
			WHERE jump_id = $1 AND t >= to_timestamp($2) AND t <= to_timestamp($3)
			ORDER BY t ASC;
			""",
			jid,
			float(b0),
			float(b1),
		)
		bias_x: list[float] = []
		bias_y: list[float] = []
		bias_z: list[float] = []
		for r in bias_rows:
			try:
				if r["gyro_x"] is not None:
					bias_x.append(float(r["gyro_x"]))
				if r["gyro_y"] is not None:
					bias_y.append(float(r["gyro_y"]))
				if r["gyro_z"] is not None:
					bias_z.append(float(r["gyro_z"]))
			except Exception:
				continue
		gx_bias = float(statistics.median(bias_x)) if bias_x else 0.0
		gy_bias = float(statistics.median(bias_y)) if bias_y else 0.0
		gz_bias = float(statistics.median(bias_z)) if bias_z else 0.0

		rows = await conn.fetch(
			"""
			SELECT EXTRACT(EPOCH FROM t) AS t_s, gyro_x, gyro_y, gyro_z
			FROM imu_samples
			WHERE jump_id = $1 AND t >= to_timestamp($2) AND t <= to_timestamp($3)
			ORDER BY t ASC;
			""",
			jid,
			float(t0),
			float(t1),
		)
		ts: list[float] = []
		gx: list[float] = []
		gy: list[float] = []
		gz: list[float] = []
		for r in rows:
			try:
				t_s = float(r["t_s"])
			except Exception:
				continue
			try:
				vx = float(r["gyro_x"]) - gx_bias if r["gyro_x"] is not None else 0.0
				vy = float(r["gyro_y"]) - gy_bias if r["gyro_y"] is not None else 0.0
				vz = float(r["gyro_z"]) - gz_bias if r["gyro_z"] is not None else 0.0
			except Exception:
				continue
			ts.append(t_s)
			gx.append(vx)
			gy.append(vy)
			gz.append(vz)

		omega: list[float] = []
		for i in range(len(ts)):
			try:
				omega.append(math.sqrt(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]))
			except Exception:
				omega.append(0.0)

		max_abs_omega = max((abs(v) for v in omega), default=0.0)
		assume_rad_s = bool(max_abs_omega < 50.0)
		deg_to_rad = math.pi / 180.0

		theta_rad = 0.0
		t_peak = None
		if len(ts) >= 2:
			for i in range(1, len(ts)):
				dt = ts[i] - ts[i - 1]
				if dt <= 0:
					continue
				w0 = omega[i - 1]
				w1 = omega[i]
				if not assume_rad_s:
					w0 *= deg_to_rad
					w1 *= deg_to_rad
				theta_rad += 0.5 * (w0 + w1) * dt
			try:
				i_peak = max(range(len(omega)), key=lambda k: abs(omega[k]))
				t_peak = ts[i_peak]
			except Exception:
				t_peak = None

		revolutions_est = abs(theta_rad) / (2.0 * math.pi) if math.pi else 0.0
		revolutions_class = int(round(revolutions_est))
		underrotation = float(revolutions_class) - float(revolutions_est)
		underrot_flag = bool(underrotation < -0.25)
		flight_time = float(t1 - t0)
		height = g_m_s2 * (flight_time ** 2) / 8.0
		phase = None
		if t_peak is not None and flight_time > 1e-6:
			phase = float((t_peak - t0) / flight_time)

		await conn.execute(
			"""
			UPDATE jumps
			SET flight_time_marked = $1,
			    height_marked = $2,
			    rotation_phase_marked = $3,
			    theta_z_rad_marked = $4,
			    revolutions_est_marked = $5,
			    revolutions_class_marked = $6,
			    underrotation_marked = $7,
			    underrot_flag_marked = $8,
			    gz_bias_marked = $9
			WHERE id = $10;
			""",
			float(flight_time),
			float(height),
			(float(phase) if phase is not None else None),
			float(theta_rad),
			float(revolutions_est),
			int(revolutions_class),
			float(underrotation),
			bool(underrot_flag),
			float(gz_bias),
			jid,
		)

		return {
			"ok": True,
			"jump_id": jid,
			"event_id": ev,
			"flight_time_marked": float(flight_time),
			"height_marked": float(height),
			"theta_z_rad_marked": float(theta_rad),
			"revolutions_est_marked": float(revolutions_est),
			"revolutions_class_marked": int(revolutions_class),
			"underrotation_marked": float(underrotation),
			"underrot_flag_marked": bool(underrot_flag),
			"gz_bias_marked": float(gz_bias),
			"rotation_phase_marked": float(phase) if phase is not None else None,
		}


async def resolve_jump_id_from_event_id(event_id: int) -> Optional[int]:
	"""
	Resolve event_id to jumps.id (jump_id). Returns the latest jump row for that event_id.
	Used by API event_id compatibility routes.
	"""
	return await resolve_jump_row_id(event_id=int(event_id))


async def update_jump_pose_metrics(
	event_id: int,
	flight_time_pose: Optional[float] = None,
	height_pose: Optional[float] = None,
	revolutions_pose: Optional[float] = None,
	pose_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Store pose-derived metrics (computed elsewhere), by event_id.
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	ev = int(event_id)
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET flight_time_pose = COALESCE($1, flight_time_pose),
			    height_pose = COALESCE($2, height_pose),
			    revolutions_pose = COALESCE($3, revolutions_pose),
			    pose_meta = COALESCE($4, pose_meta)
			WHERE event_id = $5;
			""",
			(float(flight_time_pose) if flight_time_pose is not None else None),
			(float(height_pose) if height_pose is not None else None),
			(float(revolutions_pose) if revolutions_pose is not None else None),
			pose_meta,
			ev,
		)
		return {"ok": True, "event_id": ev}


async def update_jump_pose_metrics_by_jump_id(
	jump_id: int,
	flight_time_pose: Optional[float] = None,
	height_pose: Optional[float] = None,
	revolutions_pose: Optional[float] = None,
	pose_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Store pose-derived metrics (computed elsewhere), by jumps.id (jump_id). Canonical form.
	"""
	pool = get_pool()
	if pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	jid = int(jump_id)
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET flight_time_pose = COALESCE($1, flight_time_pose),
			    height_pose = COALESCE($2, height_pose),
			    revolutions_pose = COALESCE($3, revolutions_pose),
			    pose_meta = COALESCE($4, pose_meta)
			WHERE id = $5;
			""",
			(float(flight_time_pose) if flight_time_pose is not None else None),
			(float(height_pose) if height_pose is not None else None),
			(float(revolutions_pose) if revolutions_pose is not None else None),
			pose_meta,
			jid,
		)
		return {"ok": True, "jump_id": jid}


async def list_jumps(limit: int = 200) -> List[Dict[str, Any]]:
	"""
	Return recent jumps ordered by t_peak DESC (detection time).
	"""
	pool = get_pool()
	if pool is None:
		return []
	lim = max(1, min(int(limit), 1000))
	async with pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, event_id, session_id, video_path, t_peak, t_start, t_end, flight_time, height,
			       t_takeoff_calc, t_landing_calc,
			       t_takeoff_video, t_takeoff_video_t, t_landing_video, t_landing_video_t,
			       theta_z_rad, revolutions_est, revolutions_class, underrotation, underrot_flag,
			       flight_time_marked, height_marked, rotation_phase_marked,
			       theta_z_rad_marked, revolutions_est_marked, revolutions_class_marked, underrotation_marked, underrot_flag_marked,
			       flight_time_pose, height_pose, revolutions_pose,
			       acc_peak, gyro_peak, rotation_phase, confidence,
			       name, note, created_at
			FROM jumps
			ORDER BY t_peak DESC
			LIMIT $1;
			""",
			lim,
		)
	out = [jump_row_to_dict(r, include_extra=False) for r in rows]
	return out


async def get_jump_with_imu(event_id: int) -> Optional[Dict[str, Any]]:
	"""
	Fetch one jump row and its IMU samples by event_id.
	"""
	pool = get_pool()
	if pool is None:
		return None
	async with pool.acquire() as conn:
		j = await conn.fetchrow(
			"""
			SELECT id, event_id, session_id, video_path, t_peak, t_start, t_end,
			       t_takeoff_calc, t_landing_calc,
			       t_takeoff_video, t_takeoff_video_t, t_landing_video, t_landing_video_t,
			       theta_z_rad, revolutions_est, revolutions_class, underrotation, underrot_flag,
			       flight_time_marked, height_marked, rotation_phase_marked,
			       theta_z_rad_marked, revolutions_est_marked, revolutions_class_marked, underrotation_marked, underrot_flag_marked,
			       gz_bias, gz_bias_marked,
			       flight_time_pose, height_pose, revolutions_pose, pose_meta,
			       flight_time, height, acc_peak, gyro_peak, rotation_phase, confidence,
			       name, note, created_at
			FROM jumps
			WHERE event_id = $1
			ORDER BY created_at DESC
			LIMIT 1;
			""",
			int(event_id),
		)
		if not j:
			return None
		jump_id = int(j["id"])
		samples = await conn.fetch(
			"""
			SELECT t, imu_timestamp,
			       acc_x, acc_y, acc_z,
			       gyro_x, gyro_y, gyro_z,
			       mag_x, mag_y, mag_z
			FROM imu_samples
			WHERE jump_id = $1
			ORDER BY t ASC;
			""",
			jump_id,
		)

	out_samples = [imu_sample_row_to_dict(s) for s in samples]

	# Fetch per-jump clip frame mapping (if available)
	jframes = await get_jump_frames(jump_id)

	result = jump_row_to_dict(j, include_extra=True)
	result["imu_samples"] = out_samples
	result["frames"] = jframes
	result["jump_id"] = jump_id
	return result


async def get_jump_with_imu_by_jump_id(jump_id: int) -> Optional[Dict[str, Any]]:
	"""
	Fetch one jump row and its IMU samples by jumps.id (jump_id).

	This is the preferred method because event_id is not guaranteed unique across sessions.
	"""
	pool = get_pool()
	if pool is None:
		return None
	jid = int(jump_id)
	async with pool.acquire() as conn:
		j = await conn.fetchrow(
			"""
			SELECT id, event_id, session_id, video_path, t_peak, t_start, t_end,
			       t_takeoff_calc, t_landing_calc,
			       t_takeoff_video, t_takeoff_video_t, t_landing_video, t_landing_video_t,
			       theta_z_rad, revolutions_est, revolutions_class, underrotation, underrot_flag,
			       flight_time_marked, height_marked, rotation_phase_marked,
			       theta_z_rad_marked, revolutions_est_marked, revolutions_class_marked, underrotation_marked, underrot_flag_marked,
			       gz_bias, gz_bias_marked,
			       flight_time_pose, height_pose, revolutions_pose, pose_meta,
			       flight_time, height, acc_peak, gyro_peak, rotation_phase, confidence,
			       name, note, created_at
			FROM jumps
			WHERE id = $1
			LIMIT 1;
			""",
			jid,
		)
		if not j:
			return None
		samples = await conn.fetch(
			"""
			SELECT t, imu_timestamp,
			       acc_x, acc_y, acc_z,
			       gyro_x, gyro_y, gyro_z,
			       mag_x, mag_y, mag_z
			FROM imu_samples
			WHERE jump_id = $1
			ORDER BY t ASC;
			""",
			jid,
		)

	out_samples = [imu_sample_row_to_dict(s) for s in samples]

	# Fetch per-jump clip frame mapping (if available)
	jframes = await get_jump_frames(jid)

	result = jump_row_to_dict(j, include_extra=True)
	result["imu_samples"] = out_samples
	result["frames"] = jframes
	result["jump_id"] = jid
	return result


async def resolve_jump_row_id(
	*,
	jump_id: Optional[int] = None,
	event_id: Optional[int] = None,
	session_id: Optional[str] = None,
) -> Optional[int]:
	"""
	Resolve the canonical jumps.id for a clip job.

	Why: jobs can persist across restarts, DB resets, or race the DB insert.
	This helper verifies the referenced jump_id exists; otherwise falls back to
	(event_id, session_id) or event_id lookup.
	"""
	pool = get_pool()
	if pool is None:
		return None
	jid = int(jump_id) if jump_id is not None else None
	eid = int(event_id) if event_id is not None else None
	sid = (session_id or "").strip() or None
	async with pool.acquire() as conn:
		# Prefer verifying the provided jump_id.
		if jid is not None and jid > 0:
			row = await conn.fetchrow("SELECT id FROM jumps WHERE id = $1;", jid)
			if row and row.get("id") is not None:
				return int(row["id"])

		# Fall back to session_id + event_id (more stable across DB resets for a given run).
		if eid is not None and sid is not None:
			row = await conn.fetchrow(
				"""
				SELECT id
				FROM jumps
				WHERE event_id = $1 AND session_id = $2
				ORDER BY created_at DESC
				LIMIT 1;
				""",
				int(eid),
				sid,
			)
			if row and row.get("id") is not None:
				return int(row["id"])

		# Last resort: event_id only (may be ambiguous across sessions; picks latest).
		if eid is not None:
			row = await conn.fetchrow(
				"""
				SELECT id
				FROM jumps
				WHERE event_id = $1
				ORDER BY created_at DESC
				LIMIT 1;
				""",
				int(eid),
			)
			if row and row.get("id") is not None:
				return int(row["id"])
	return None


async def set_jump_video_path(event_id: int, video_path: Optional[str]) -> None:
	"""
	Set the per-jump clip path for a given event_id.
	"""
	pool = get_pool()
	if pool is None:
		return
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET video_path = $1
			WHERE event_id = $2;
			""",
			video_path,
			int(event_id),
		)


async def set_jump_video_path_by_jump_id(jump_id: int, video_path: Optional[str]) -> None:
	"""
	Set the per-jump clip path for a given internal jump_id (preferred over event_id).
	"""
	pool = get_pool()
	if pool is None:
		return
	async with pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE jumps
			SET video_path = $1
			WHERE id = $2;
			""",
			video_path,
			int(jump_id),
		)


async def delete_jump(event_id: int) -> Dict[str, Any]:
	"""
	Delete the *most recently created* jump row for a given event_id, along with
	all associated IMU samples.

	Notes:
	- `event_id` is not enforced as unique in the schema, so we delete the latest
	  row by `created_at DESC`.
	- `imu_samples.jump_id` and `jump_frames.jump_id` have `ON DELETE CASCADE`, so deleting the jump row
	  deletes its samples and frames automatically.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] delete_jump: pool is None, skipping")
		return {"deleted": False, "detail": "DB not configured"}

	eid = int(event_id)
	async with pool.acquire() as conn:
		async with conn.transaction():
			row = await conn.fetchrow(
				"""
				SELECT id
				FROM jumps
				WHERE event_id = $1
				ORDER BY created_at DESC
				LIMIT 1;
				""",
				eid,
			)
			if not row:
				return {"deleted": False, "detail": f"No jump found for event_id={eid}"}
			jump_id = int(row["id"])
			imu_cnt = await conn.fetchval("SELECT COUNT(*) FROM imu_samples WHERE jump_id = $1;", jump_id)
			frame_cnt = await conn.fetchval("SELECT COUNT(*) FROM jump_frames WHERE jump_id = $1;", jump_id)
			del_row = await conn.fetchrow(
				"""
				DELETE FROM jumps
				WHERE id = $1
				RETURNING id, event_id;
				""",
				jump_id,
			)
			return {
				"deleted": bool(del_row),
				"jump_id": jump_id,
				"event_id": eid,
				"imu_samples_deleted": int(imu_cnt or 0),
				"frames_deleted": int(frame_cnt or 0),
				"detail": f"Deleted jump event_id={eid} (jump_id={jump_id}), {int(imu_cnt or 0)} IMU samples, and {int(frame_cnt or 0)} frames",
			}


async def delete_jump_by_jump_id(jump_id: int) -> Dict[str, Any]:
	"""
	Delete a specific jump row by jumps.id (jump_id), along with its IMU samples.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] delete_jump_by_jump_id: pool is None, skipping")
		return {"deleted": False, "detail": "DB not configured"}
	jid = int(jump_id)
	async with pool.acquire() as conn:
		async with conn.transaction():
			row = await conn.fetchrow(
				"""
				SELECT id, event_id
				FROM jumps
				WHERE id = $1
				LIMIT 1;
				""",
				jid,
			)
			if not row:
				return {"deleted": False, "detail": f"No jump found for jump_id={jid}"}
			eid = row["event_id"]
			imu_cnt = await conn.fetchval("SELECT COUNT(*) FROM imu_samples WHERE jump_id = $1;", jid)
			frame_cnt = await conn.fetchval("SELECT COUNT(*) FROM jump_frames WHERE jump_id = $1;", jid)
			del_row = await conn.fetchrow(
				"""
				DELETE FROM jumps
				WHERE id = $1
				RETURNING id, event_id;
				""",
				jid,
			)
			return {
				"deleted": bool(del_row),
				"jump_id": jid,
				"event_id": eid,
				"imu_samples_deleted": int(imu_cnt or 0),
				"frames_deleted": int(frame_cnt or 0),
				"detail": f"Deleted jump jump_id={jid} (event_id={eid}), {int(imu_cnt or 0)} IMU samples, and {int(frame_cnt or 0)} frames",
			}


async def delete_jumps_bulk(jump_ids: List[int]) -> Dict[str, Any]:
	"""
	Delete multiple jumps by their jump_id values, along with all associated IMU samples and frame data.
	
	Returns summary of what was deleted.
	"""
	pool = get_pool()
	if pool is None:
		logger.debug("[DB] delete_jumps_bulk: pool is None, skipping")
		return {"deleted_count": 0, "detail": "DB not configured"}
	
	if not jump_ids:
		return {"deleted_count": 0, "detail": "No jump IDs provided"}
	
	# Convert to integers and filter out invalid values
	jids = [int(jid) for jid in jump_ids if isinstance(jid, (int, str)) and str(jid).strip()]
	if not jids:
		return {"deleted_count": 0, "detail": "No valid jump IDs provided"}
	
	async with pool.acquire() as conn:
		async with conn.transaction():
			# Get counts before deletion
			placeholders = ','.join([f'${i+1}' for i in range(len(jids))])
			imu_cnt = await conn.fetchval(
				f"SELECT COUNT(*) FROM imu_samples WHERE jump_id IN ({placeholders});",
				*jids
			)
			frame_cnt = await conn.fetchval(
				f"SELECT COUNT(*) FROM jump_frames WHERE jump_id IN ({placeholders});",
				*jids
			)
			
			# Get jump info before deletion
			rows = await conn.fetch(
				f"""
				SELECT id, event_id
				FROM jumps
				WHERE id IN ({placeholders});
				""",
				*jids
			)
			
			# Delete jumps (CASCADE will handle imu_samples and jump_frames automatically)
			deleted_rows = await conn.fetch(
				f"""
				DELETE FROM jumps
				WHERE id IN ({placeholders})
				RETURNING id, event_id;
				""",
				*jids
			)
			
			deleted_count = len(deleted_rows)
			event_ids = [r["event_id"] for r in deleted_rows]
			
			return {
				"deleted_count": deleted_count,
				"jump_ids": [int(r["id"]) for r in deleted_rows],
				"event_ids": event_ids,
				"imu_samples_deleted": int(imu_cnt or 0),
				"frames_deleted": int(frame_cnt or 0),
				"detail": f"Deleted {deleted_count} jump(s), {int(imu_cnt or 0)} IMU sample(s), and {int(frame_cnt or 0)} frame(s)",
			}


# Device management functions
