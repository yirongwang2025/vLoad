"""
Simple async PostgreSQL integration for jump and IMU storage.

We use `asyncpg` directly (no full ORM) to:
  - create tables on startup if they do not exist
  - insert one jump row + associated IMU samples for that jump

The database is configured via the `DATABASE_URL` environment variable, e.g.:
  export DATABASE_URL="postgresql://user:password@localhost:5432/vload"
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import asyncpg

_pool: Optional[asyncpg.Pool] = None

# Prevent concurrent init_db calls from racing.
_init_lock = asyncio.Lock()
_warned_no_dsn: bool = False
async def init_db() -> None:
	"""
	Initialise PostgreSQL connection pool and ensure tables exist.

	If DATABASE_URL is not set, this becomes a no‑op so the app can run
	without a database (you'll just see log messages).
	"""
	global _pool
	global _warned_no_dsn
	dsn = os.getenv("DATABASE_URL") or ""
	if not dsn:
		# Running without DB is allowed. Log once so it's obvious why inserts are skipped.
		if not _warned_no_dsn:
			print("[DB] DATABASE_URL not set; persistence disabled.")
			_warned_no_dsn = True
		return

	async with _init_lock:
		if _pool is None:
			_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)

	async with _pool.acquire() as conn:
		# Create jumps table
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS jumps (
				id            SERIAL PRIMARY KEY,
				event_id      INTEGER,
				session_id    TEXT,
				t_peak        TIMESTAMPTZ NOT NULL,
				t_start       TIMESTAMPTZ,
				t_end         TIMESTAMPTZ,
				flight_time   DOUBLE PRECISION,
				height        DOUBLE PRECISION,
				acc_peak      DOUBLE PRECISION,
				gyro_peak     DOUBLE PRECISION,
				rotation_phase DOUBLE PRECISION,
				confidence    DOUBLE PRECISION,
				name          TEXT,
				note          TEXT,
				created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)

		# Backfill/upgrade existing tables (safe no-op on fresh DBs)
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS session_id TEXT;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS video_path TEXT;")

		# Create IMU samples table
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS imu_samples (
				id            BIGSERIAL PRIMARY KEY,
				jump_id       INTEGER REFERENCES jumps(id) ON DELETE CASCADE,
				t             TIMESTAMPTZ NOT NULL,
				imu_timestamp BIGINT,
				acc_x         DOUBLE PRECISION,
				acc_y         DOUBLE PRECISION,
				acc_z         DOUBLE PRECISION,
				gyro_x        DOUBLE PRECISION,
				gyro_y        DOUBLE PRECISION,
				gyro_z        DOUBLE PRECISION,
				mag_x         DOUBLE PRECISION,
				mag_y         DOUBLE PRECISION,
				mag_z         DOUBLE PRECISION
			);
			"""
		)

		# Sessions table: mirrors session.json (but keeps video on filesystem)
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS sessions (
				session_id   TEXT PRIMARY KEY,
				t_start      TIMESTAMPTZ,
				t_stop       TIMESTAMPTZ,
				imu_mode     TEXT,
				imu_rate     INTEGER,
				video_fps    INTEGER,
				video_path   TEXT,
				jump_config  JSONB,
				meta         JSONB,
				created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)

		# Frames table: mirrors frames.csv
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS frames (
				id         BIGSERIAL PRIMARY KEY,
				session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
				frame_idx  INTEGER NOT NULL,
				t_host     DOUBLE PRECISION NOT NULL,
				device_ts  DOUBLE PRECISION,
				width      INTEGER,
				height     INTEGER
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS frames_session_idx ON frames(session_id, frame_idx);")
		await conn.execute("CREATE INDEX IF NOT EXISTS frames_session_thost ON frames(session_id, t_host);")
		await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS frames_session_frame_unique ON frames(session_id, frame_idx);")

		# Jump frames table: per-jump timing mapping for clip playback sync.
		# This aligns frames to a specific jump (same as imu_samples) so jump review can
		# sync video<->IMU without relying on session-level files.
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS jump_frames (
				id         BIGSERIAL PRIMARY KEY,
				jump_id    INTEGER REFERENCES jumps(id) ON DELETE CASCADE,
				frame_idx  INTEGER NOT NULL,
				t_video    DOUBLE PRECISION NOT NULL,
				t_host     DOUBLE PRECISION NOT NULL,
				device_ts  DOUBLE PRECISION,
				width      INTEGER,
				height     INTEGER
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS jump_frames_jump_idx ON jump_frames(jump_id, frame_idx);")
		await conn.execute("CREATE INDEX IF NOT EXISTS jump_frames_jump_tvideo ON jump_frames(jump_id, t_video);")
		await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS jump_frames_jump_frame_unique ON jump_frames(jump_id, frame_idx);")


def _to_dt(sec: float) -> datetime:
	"""Convert a wall‑clock seconds float to timezone‑aware datetime."""
	return datetime.fromtimestamp(sec, tz=timezone.utc)


async def upsert_session_start(
	session_id: str,
	t_start: float,
	imu_mode: Optional[str],
	imu_rate: Optional[int],
	jump_config: Optional[Dict[str, Any]],
	video_fps: Optional[int],
	video_path: Optional[str] = None,
	meta: Optional[Dict[str, Any]] = None,
) -> None:
	"""
	Create/update a session row at session start.
	"""
	global _pool
	if _pool is None:
		print("[DB] upsert_session_start: _pool is None, skipping")
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with _pool.acquire() as conn:
		await conn.execute(
			"""
			INSERT INTO sessions (session_id, t_start, imu_mode, imu_rate, video_fps, video_path, jump_config, meta)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
			ON CONFLICT (session_id) DO UPDATE SET
				t_start = EXCLUDED.t_start,
				imu_mode = EXCLUDED.imu_mode,
				imu_rate = EXCLUDED.imu_rate,
				video_fps = EXCLUDED.video_fps,
				video_path = COALESCE(EXCLUDED.video_path, sessions.video_path),
				jump_config = EXCLUDED.jump_config,
				meta = EXCLUDED.meta;
			""",
			sid,
			_to_dt(float(t_start)),
			imu_mode,
			int(imu_rate) if imu_rate is not None else None,
			int(video_fps) if video_fps is not None else None,
			video_path,
			jump_config,
			meta,
		)


async def update_session_stop(session_id: str, t_stop: float, video_path: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
	"""
	Update t_stop and optionally video_path/meta for a session.
	"""
	global _pool
	if _pool is None:
		print("[DB] update_session_stop: _pool is None, skipping")
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with _pool.acquire() as conn:
		await conn.execute(
			"""
			UPDATE sessions
			SET t_stop = $2,
			    video_path = COALESCE($3, video_path),
			    meta = COALESCE($4, meta)
			WHERE session_id = $1;
			""",
			sid,
			_to_dt(float(t_stop)),
			video_path,
			meta,
		)


async def replace_frames(session_id: str, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Replace all frames for a session (delete then bulk insert).
	"""
	global _pool
	if _pool is None:
		print("[DB] replace_frames: _pool is None, skipping")
		return {"inserted": 0}
	sid = (session_id or "").strip()
	if not sid:
		return {"inserted": 0}
	async with _pool.acquire() as conn:
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
				columns=["session_id", "frame_idx", "t_host", "device_ts", "width", "height"],
			)
			return {"inserted": len(records)}


async def get_frames(session_id: str, limit: int = 200000, t0: Optional[float] = None, t1: Optional[float] = None) -> List[Dict[str, Any]]:
	"""
	Return frames for a session from DB. Optional host-time filtering.
	"""
	global _pool
	if _pool is None:
		return []
	sid = (session_id or "").strip()
	if not sid:
		return []
	lim = max(1, min(int(limit), 500000))
	async with _pool.acquire() as conn:
		if t0 is not None and t1 is not None:
			rows = await conn.fetch(
				"""
				SELECT frame_idx, t_host, device_ts, width, height
				FROM frames
				WHERE session_id = $1 AND t_host >= $2 AND t_host <= $3
				ORDER BY frame_idx ASC
				LIMIT $4;
				""",
				sid,
				float(t0),
				float(t1),
				lim,
			)
		else:
			rows = await conn.fetch(
				"""
				SELECT frame_idx, t_host, device_ts, width, height
				FROM frames
				WHERE session_id = $1
				ORDER BY frame_idx ASC
				LIMIT $2;
				""",
				sid,
				lim,
			)
	out: List[Dict[str, Any]] = []
	for r in rows:
		out.append(
			{
				"frame_idx": int(r["frame_idx"]),
				"t_host": float(r["t_host"]),
				"device_ts": float(r["device_ts"]) if r["device_ts"] is not None else None,
				"width": int(r["width"]) if r["width"] is not None else None,
				"height": int(r["height"]) if r["height"] is not None else None,
			}
		)
	return out


async def replace_jump_frames(jump_id: int, frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Replace all frames for a jump (delete then bulk insert).
	Frames are expected to be clip-relative: frame_idx starts at 0, t_video starts at 0.
	"""
	global _pool
	if _pool is None:
		print("[DB] replace_jump_frames: _pool is None, skipping")
		return {"inserted": 0}
	jid = int(jump_id)
	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
		return []
	jid = int(jump_id)
	lim = max(1, min(int(limit), 500000))
	async with _pool.acquire() as conn:
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
	out: List[Dict[str, Any]] = []
	for r in rows:
		out.append(
			{
				"frame_idx": int(r["frame_idx"]),
				"t_video": float(r["t_video"]),
				"t_host": float(r["t_host"]),
				"device_ts": float(r["device_ts"]) if r["device_ts"] is not None else None,
				"width": int(r["width"]) if r["width"] is not None else None,
				"height": int(r["height"]) if r["height"] is not None else None,
			}
		)
	return out


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
	global _pool
	if _pool is None:
		print("[DB] insert_jump_with_imu: _pool is None, skipping")
		return None
	print(f"[DB] insert_jump_with_imu: event_id={jump.get('event_id')}, samples={len(imu_samples)}")

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

	async with _pool.acquire() as conn:
		# Insert into jumps table and get generated id
		jump_id = await conn.fetchval(
			"""
			INSERT INTO jumps (
				event_id, session_id, video_path, t_peak, t_start, t_end,
				flight_time, height, acc_peak, gyro_peak,
				rotation_phase, confidence, name, note
			)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
			RETURNING id;
			""",
			event_id,
			session_id,
			video_path,
			t_peak_dt,
			t_start_dt,
			t_end_dt,
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
	global _pool
	if _pool is None:
		print("[DB] update_annotation: _pool is None, skipping")
		return
	print(f"[DB] update_annotation: event_id={event_id}, name={name!r}")

	async with _pool.acquire() as conn:
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


async def list_jumps(limit: int = 200) -> List[Dict[str, Any]]:
	"""
	Return recent jumps ordered by t_peak DESC (detection time).
	"""
	global _pool
	if _pool is None:
		return []
	lim = max(1, min(int(limit), 1000))
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, event_id, session_id, video_path, t_peak, t_start, t_end, flight_time, height,
			       acc_peak, gyro_peak, rotation_phase, confidence,
			       name, note, created_at
			FROM jumps
			ORDER BY t_peak DESC
			LIMIT $1;
			""",
			lim,
		)
	out: List[Dict[str, Any]] = []
	for r in rows:
		out.append(
			{
				"jump_id": int(r["id"]),
				"event_id": r["event_id"],
				"session_id": r["session_id"],
				"video_path": r["video_path"],
				"t_peak": r["t_peak"].timestamp() if r["t_peak"] else None,
				"t_start": r["t_start"].timestamp() if r["t_start"] else None,
				"t_end": r["t_end"].timestamp() if r["t_end"] else None,
				"flight_time": r["flight_time"],
				"height": r["height"],
				"acc_peak": r["acc_peak"],
				"gyro_peak": r["gyro_peak"],
				"rotation_phase": r["rotation_phase"],
				"confidence": r["confidence"],
				"name": r["name"],
				"note": r["note"],
				"created_at": r["created_at"].timestamp() if r["created_at"] else None,
			}
		)
	return out


async def get_jump_with_imu(event_id: int) -> Optional[Dict[str, Any]]:
	"""
	Fetch one jump row and its IMU samples by event_id.
	"""
	global _pool
	if _pool is None:
		return None
	async with _pool.acquire() as conn:
		j = await conn.fetchrow(
			"""
			SELECT id, event_id, session_id, video_path, t_peak, t_start, t_end,
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

	out_samples: List[Dict[str, Any]] = []
	for s in samples:
		out_samples.append(
			{
				"t": s["t"].timestamp() if s["t"] else None,
				"imu_timestamp": s["imu_timestamp"],
				"acc": [s["acc_x"], s["acc_y"], s["acc_z"]],
				"gyro": [s["gyro_x"], s["gyro_y"], s["gyro_z"]],
				"mag": [s["mag_x"], s["mag_y"], s["mag_z"]],
			}
		)

	# Fetch per-jump clip frame mapping (if available)
	jframes = await get_jump_frames(jump_id)

	return {
		"jump_id": jump_id,
		"event_id": j["event_id"],
		"session_id": j["session_id"],
		"video_path": j["video_path"],
		"t_peak": j["t_peak"].timestamp() if j["t_peak"] else None,
		"t_start": j["t_start"].timestamp() if j["t_start"] else None,
		"t_end": j["t_end"].timestamp() if j["t_end"] else None,
		"flight_time": j["flight_time"],
		"height": j["height"],
		"acc_peak": j["acc_peak"],
		"gyro_peak": j["gyro_peak"],
		"rotation_phase": j["rotation_phase"],
		"confidence": j["confidence"],
		"name": j["name"],
		"note": j["note"],
		"created_at": j["created_at"].timestamp() if j["created_at"] else None,
		"imu_samples": out_samples,
		"frames": jframes,
	}


async def set_jump_video_path(event_id: int, video_path: Optional[str]) -> None:
	"""
	Set the per-jump clip path for a given event_id.
	"""
	global _pool
	if _pool is None:
		return
	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
		return
	async with _pool.acquire() as conn:
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
	- `imu_samples.jump_id` has `ON DELETE CASCADE`, so deleting the jump row
	  deletes its samples automatically.
	"""
	global _pool
	if _pool is None:
		print("[DB] delete_jump: _pool is None, skipping")
		return {"deleted": False, "detail": "DB not configured"}

	eid = int(event_id)
	async with _pool.acquire() as conn:
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
				"detail": f"Deleted jump event_id={eid} (jump_id={jump_id}) and {int(imu_cnt or 0)} IMU samples",
			}


async def close_db() -> None:
	"""Close the connection pool on shutdown."""
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None


