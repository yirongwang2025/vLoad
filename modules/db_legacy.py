"""
Simple async PostgreSQL integration for jump and IMU storage.

We use `asyncpg` directly (no full ORM) to:
  - create tables on startup if they do not exist
  - insert one jump row + associated IMU samples for that jump

The database is configured via `config.json` (see `config.example.json`), e.g.:
  "database": { "url": "postgresql://user:password@localhost:5432/vload" }
"""

import asyncio
from datetime import datetime, timezone
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

import asyncpg

from modules.config import get_config

_pool: Optional[asyncpg.Pool] = None
_last_init_error: Optional[str] = None

# Prevent concurrent init_db calls from racing.
_init_lock = asyncio.Lock()
_warned_no_dsn: bool = False
async def init_db() -> None:
	"""
	Initialise PostgreSQL connection pool and ensure tables exist.

	If database.url is not set in config.json, this becomes a no‑op so the app can run
	without a database (you'll just see log messages).
	"""
	global _pool
	global _warned_no_dsn
	global _last_init_error
	dsn = (get_config().database.url or "").strip()
	if not dsn:
		# Running without DB is allowed. Log once so it's obvious why inserts are skipped.
		if not _warned_no_dsn:
			print("[DB] database.url not set in config.json; persistence disabled.")
			_warned_no_dsn = True
		_last_init_error = "database.url not set"
		return

	async with _init_lock:
		if _pool is None:
			try:
				_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
				_last_init_error = None
			except Exception as e:
				_last_init_error = repr(e)
				raise

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
				-- Algorithm-estimated takeoff/landing (host time)
				t_takeoff_calc   TIMESTAMPTZ,
				t_landing_calc   TIMESTAMPTZ,
				-- Video-verified takeoff/landing (host time + clip-relative video time)
				t_takeoff_video  TIMESTAMPTZ,
				t_landing_video  TIMESTAMPTZ,
				t_takeoff_video_t DOUBLE PRECISION,
				t_landing_video_t DOUBLE PRECISION,
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
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_takeoff_calc TIMESTAMPTZ;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_landing_calc TIMESTAMPTZ;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_takeoff_video TIMESTAMPTZ;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_landing_video TIMESTAMPTZ;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_takeoff_video_t DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS t_landing_video_t DOUBLE PRECISION;")

		# --- Original algorithm metrics (beyond height/flight_time) ---
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS theta_z_rad DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS revolutions_est DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS revolutions_class INTEGER;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS underrotation DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS underrot_flag BOOLEAN;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS gz_bias DOUBLE PRECISION;")

		# --- IMU metrics computed using *video-verified* takeoff/landing marks ---
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS flight_time_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS height_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS rotation_phase_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS theta_z_rad_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS revolutions_est_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS revolutions_class_marked INTEGER;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS underrotation_marked DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS underrot_flag_marked BOOLEAN;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS gz_bias_marked DOUBLE PRECISION;")

		# --- Video pose analysis metrics (stored when available) ---
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS flight_time_pose DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS height_pose DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS revolutions_pose DOUBLE PRECISION;")
		await conn.execute("ALTER TABLE jumps ADD COLUMN IF NOT EXISTS pose_meta JSONB;")

		# Dedupe and enforce idempotency:
		# event_id is not globally unique (it resets per session), but (session_id, event_id) SHOULD be.
		# If prior versions inserted duplicates, clean them up before creating a UNIQUE index.
		try:
			await conn.execute(
				"""
				WITH ranked AS (
					SELECT id,
						   ROW_NUMBER() OVER (
							   PARTITION BY session_id, event_id
							   ORDER BY
							     (name IS NOT NULL AND name <> '') DESC,
							     (note IS NOT NULL AND note <> '') DESC,
							     (t_takeoff_video IS NOT NULL) DESC,
							     (t_landing_video IS NOT NULL) DESC,
							     (video_path IS NOT NULL AND video_path <> '') DESC,
							     created_at DESC
						   ) AS rn
					FROM jumps
					WHERE session_id IS NOT NULL AND event_id IS NOT NULL
				)
				DELETE FROM jumps
				WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
				"""
			)
		except Exception as e:
			print(f"[DB] Warning: dedupe jumps failed: {e!r}")

		# Enforce uniqueness for detected events within a session (prevents duplicate rows from retries).
		try:
			await conn.execute(
				"CREATE UNIQUE INDEX IF NOT EXISTS jumps_session_event_unique ON jumps(session_id, event_id);"
			)
		except Exception as e:
			# If duplicates remain, this can still fail; log and continue so the app can run.
			print(f"[DB] Warning: failed to create UNIQUE index jumps_session_event_unique: {e!r}")

		# Helpful read/index patterns
		await conn.execute("CREATE INDEX IF NOT EXISTS jumps_session_tpeak_idx ON jumps(session_id, t_peak DESC);")

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
				video_backend TEXT,
				video_fps    INTEGER,
				video_path   TEXT,
				camera_clock_offset_s DOUBLE PRECISION,
				jump_config  JSONB,
				meta         JSONB,
				created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)
		await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS video_backend TEXT;")
		await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS camera_clock_offset_s DOUBLE PRECISION;")

		# Frames table: mirrors frames.csv
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS frames (
				id         BIGSERIAL PRIMARY KEY,
				session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
				frame_idx  INTEGER NOT NULL,
				t_host     DOUBLE PRECISION NOT NULL,
				device_ts  DOUBLE PRECISION,
				source     TEXT,
				width      INTEGER,
				height     INTEGER
			);
			"""
		)
		await conn.execute("ALTER TABLE frames ADD COLUMN IF NOT EXISTS source TEXT;")
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

		# Devices table: store MAC address -> name mappings for IMU sensors
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS devices (
				id          BIGSERIAL PRIMARY KEY,
				mac_address TEXT NOT NULL UNIQUE,
				name        TEXT NOT NULL,
				created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS devices_mac_idx ON devices(mac_address);")
		await conn.execute("CREATE INDEX IF NOT EXISTS devices_name_idx ON devices(name);")
		
		# Add trigger to update updated_at timestamp
		await conn.execute("""
			CREATE OR REPLACE FUNCTION update_devices_updated_at()
			RETURNS TRIGGER AS $$
			BEGIN
				NEW.updated_at = NOW();
				RETURN NEW;
			END;
			$$ LANGUAGE plpgsql;
		""")
		await conn.execute("""
			DROP TRIGGER IF EXISTS devices_updated_at_trigger ON devices;
			CREATE TRIGGER devices_updated_at_trigger
			BEFORE UPDATE ON devices
			FOR EACH ROW
			EXECUTE FUNCTION update_devices_updated_at();
		""")

		# Skaters table: store skater profiles
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS skaters (
				id          BIGSERIAL PRIMARY KEY,
				name        TEXT NOT NULL,
				date_of_birth DATE,
				gender      TEXT,
				level       TEXT,
				club        TEXT,
				email       TEXT,
				phone       TEXT,
				notes       TEXT,
				created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS skaters_name_idx ON skaters(name);")
		await conn.execute("CREATE INDEX IF NOT EXISTS skaters_level_idx ON skaters(level);")
		
		# Coaches table: store coach profiles
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS coaches (
				id          BIGSERIAL PRIMARY KEY,
				name        TEXT NOT NULL,
				email       TEXT,
				phone       TEXT,
				certification TEXT,
				level       TEXT,
				club        TEXT,
				notes       TEXT,
				created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS coaches_name_idx ON coaches(name);")
		await conn.execute("CREATE INDEX IF NOT EXISTS coaches_level_idx ON coaches(level);")
		
		# Add triggers for updated_at on skaters and coaches
		await conn.execute("""
			DROP TRIGGER IF EXISTS skaters_updated_at_trigger ON skaters;
			CREATE TRIGGER skaters_updated_at_trigger
			BEFORE UPDATE ON skaters
			FOR EACH ROW
			EXECUTE FUNCTION update_devices_updated_at();
		""")
		await conn.execute("""
			DROP TRIGGER IF EXISTS coaches_updated_at_trigger ON coaches;
			CREATE TRIGGER coaches_updated_at_trigger
			BEFORE UPDATE ON coaches
			FOR EACH ROW
			EXECUTE FUNCTION update_devices_updated_at();
		""")

		# Skater-Coach relationships: many-to-many with head coach flag
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS skater_coaches (
				id            BIGSERIAL PRIMARY KEY,
				skater_id     INTEGER NOT NULL REFERENCES skaters(id) ON DELETE CASCADE,
				coach_id      INTEGER NOT NULL REFERENCES coaches(id) ON DELETE CASCADE,
				is_head_coach BOOLEAN NOT NULL DEFAULT FALSE,
				created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				UNIQUE(skater_id, coach_id)
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS skater_coaches_skater_idx ON skater_coaches(skater_id);")
		await conn.execute("CREATE INDEX IF NOT EXISTS skater_coaches_coach_idx ON skater_coaches(coach_id);")

		# Skater-Device relationships: many-to-many with placement info
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS skater_devices (
				id         BIGSERIAL PRIMARY KEY,
				skater_id  INTEGER NOT NULL REFERENCES skaters(id) ON DELETE CASCADE,
				device_id  INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
				placement  TEXT NOT NULL DEFAULT 'waist',
				created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				UNIQUE(skater_id, device_id)
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS skater_devices_skater_idx ON skater_devices(skater_id);")
		await conn.execute("CREATE INDEX IF NOT EXISTS skater_devices_device_idx ON skater_devices(device_id);")

		# Skater detection settings: store per-skater jump detection parameters
		await conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS skater_detection_settings (
				id                        BIGSERIAL PRIMARY KEY,
				skater_id                 INTEGER NOT NULL UNIQUE REFERENCES skaters(id) ON DELETE CASCADE,
				min_jump_height_m          REAL,
				min_jump_peak_az_no_g     REAL,
				min_jump_peak_gz_deg_s    REAL,
				min_new_event_separation_s REAL,
				min_revs                  REAL,
				analysis_interval_s       REAL,
				created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
				updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
			);
			"""
		)
		await conn.execute("CREATE INDEX IF NOT EXISTS skater_detection_settings_skater_idx ON skater_detection_settings(skater_id);")


def get_status() -> Dict[str, Any]:
	"""
	Return lightweight DB status for diagnostics (used by /db/status).
	"""
	dsn = (get_config().database.url or "").strip()
	return {
		"enabled": bool(dsn),
		"dsn_set": bool(dsn),
		"pool_ready": _pool is not None,
		"last_init_error": _last_init_error,
	}


def _to_dt(sec: float) -> datetime:
	"""Convert a wall‑clock seconds float to timezone‑aware datetime."""
	return datetime.fromtimestamp(sec, tz=timezone.utc)


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


async def update_session_camera_calibration(session_id: str, camera_clock_offset_s: Optional[float], video_backend: Optional[str] = None) -> None:
	"""
	Store a simple timebase mapping hint so future camera backends (e.g. Jetson/GStreamer)
	can cleanly map device timestamps to host timestamps.
	"""
	global _pool
	if _pool is None:
		return
	sid = (session_id or "").strip()
	if not sid:
		return
	async with _pool.acquire() as conn:
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
				SELECT frame_idx, t_host, device_ts, source, width, height
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
				SELECT frame_idx, t_host, device_ts, source, width, height
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
				"source": str(r["source"]) if r.get("source") is not None else None,
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
	print(f"[DB] insert_jump_with_imu: session_id={jump.get('session_id')}, event_id={jump.get('event_id')}, samples={len(imu_samples)}")

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

	async with _pool.acquire() as conn:
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


async def update_annotation_by_jump_id(jump_id: int, name: Optional[str], note: Optional[str]) -> None:
	"""
	Update the name/note annotation for a jump row identified by jumps.id (jump_id).

	This is the preferred method because event_id is not guaranteed unique across sessions.
	"""
	global _pool
	if _pool is None:
		print("[DB] update_annotation_by_jump_id: _pool is None, skipping")
		return
	jid = int(jump_id)
	print(f"[DB] update_annotation_by_jump_id: jump_id={jid}, name={name!r}")

	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
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

	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
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

	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	ev = int(event_id)
	g_m_s2 = 9.80665

	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	jid = int(jump_id)
	g_m_s2 = 9.80665

	async with _pool.acquire() as conn:
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


async def update_jump_pose_metrics(
	event_id: int,
	flight_time_pose: Optional[float] = None,
	height_pose: Optional[float] = None,
	revolutions_pose: Optional[float] = None,
	pose_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Store pose-derived metrics (computed elsewhere).
	"""
	global _pool
	if _pool is None:
		return {"ok": False, "error": "DB not configured", "disabled": True}
	ev = int(event_id)
	async with _pool.acquire() as conn:
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
				"t_takeoff_calc": r["t_takeoff_calc"].timestamp() if r.get("t_takeoff_calc") else None,
				"t_landing_calc": r["t_landing_calc"].timestamp() if r.get("t_landing_calc") else None,
				"t_takeoff_video": r["t_takeoff_video"].timestamp() if r.get("t_takeoff_video") else None,
				"t_takeoff_video_t": float(r["t_takeoff_video_t"]) if r.get("t_takeoff_video_t") is not None else None,
				"t_landing_video": r["t_landing_video"].timestamp() if r.get("t_landing_video") else None,
				"t_landing_video_t": float(r["t_landing_video_t"]) if r.get("t_landing_video_t") is not None else None,
				"theta_z_rad": float(r["theta_z_rad"]) if r.get("theta_z_rad") is not None else None,
				"revolutions_est": float(r["revolutions_est"]) if r.get("revolutions_est") is not None else None,
				"revolutions_class": int(r["revolutions_class"]) if r.get("revolutions_class") is not None else None,
				"underrotation": float(r["underrotation"]) if r.get("underrotation") is not None else None,
				"underrot_flag": bool(r["underrot_flag"]) if r.get("underrot_flag") is not None else None,
				"flight_time_marked": float(r["flight_time_marked"]) if r.get("flight_time_marked") is not None else None,
				"height_marked": float(r["height_marked"]) if r.get("height_marked") is not None else None,
				"rotation_phase_marked": float(r["rotation_phase_marked"]) if r.get("rotation_phase_marked") is not None else None,
				"theta_z_rad_marked": float(r["theta_z_rad_marked"]) if r.get("theta_z_rad_marked") is not None else None,
				"revolutions_est_marked": float(r["revolutions_est_marked"]) if r.get("revolutions_est_marked") is not None else None,
				"revolutions_class_marked": int(r["revolutions_class_marked"]) if r.get("revolutions_class_marked") is not None else None,
				"underrotation_marked": float(r["underrotation_marked"]) if r.get("underrotation_marked") is not None else None,
				"underrot_flag_marked": bool(r["underrot_flag_marked"]) if r.get("underrot_flag_marked") is not None else None,
				"flight_time_pose": float(r["flight_time_pose"]) if r.get("flight_time_pose") is not None else None,
				"height_pose": float(r["height_pose"]) if r.get("height_pose") is not None else None,
				"revolutions_pose": float(r["revolutions_pose"]) if r.get("revolutions_pose") is not None else None,
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
		"t_takeoff_calc": j["t_takeoff_calc"].timestamp() if j.get("t_takeoff_calc") else None,
		"t_landing_calc": j["t_landing_calc"].timestamp() if j.get("t_landing_calc") else None,
		"t_takeoff_video": j["t_takeoff_video"].timestamp() if j.get("t_takeoff_video") else None,
		"t_takeoff_video_t": float(j["t_takeoff_video_t"]) if j.get("t_takeoff_video_t") is not None else None,
		"t_landing_video": j["t_landing_video"].timestamp() if j.get("t_landing_video") else None,
		"t_landing_video_t": float(j["t_landing_video_t"]) if j.get("t_landing_video_t") is not None else None,
		"theta_z_rad": float(j["theta_z_rad"]) if j.get("theta_z_rad") is not None else None,
		"revolutions_est": float(j["revolutions_est"]) if j.get("revolutions_est") is not None else None,
		"revolutions_class": int(j["revolutions_class"]) if j.get("revolutions_class") is not None else None,
		"underrotation": float(j["underrotation"]) if j.get("underrotation") is not None else None,
		"underrot_flag": bool(j["underrot_flag"]) if j.get("underrot_flag") is not None else None,
		"gz_bias": float(j["gz_bias"]) if j.get("gz_bias") is not None else None,
		"flight_time_marked": float(j["flight_time_marked"]) if j.get("flight_time_marked") is not None else None,
		"height_marked": float(j["height_marked"]) if j.get("height_marked") is not None else None,
		"rotation_phase_marked": float(j["rotation_phase_marked"]) if j.get("rotation_phase_marked") is not None else None,
		"theta_z_rad_marked": float(j["theta_z_rad_marked"]) if j.get("theta_z_rad_marked") is not None else None,
		"revolutions_est_marked": float(j["revolutions_est_marked"]) if j.get("revolutions_est_marked") is not None else None,
		"revolutions_class_marked": int(j["revolutions_class_marked"]) if j.get("revolutions_class_marked") is not None else None,
		"underrotation_marked": float(j["underrotation_marked"]) if j.get("underrotation_marked") is not None else None,
		"underrot_flag_marked": bool(j["underrot_flag_marked"]) if j.get("underrot_flag_marked") is not None else None,
		"gz_bias_marked": float(j["gz_bias_marked"]) if j.get("gz_bias_marked") is not None else None,
		"flight_time_pose": float(j["flight_time_pose"]) if j.get("flight_time_pose") is not None else None,
		"height_pose": float(j["height_pose"]) if j.get("height_pose") is not None else None,
		"revolutions_pose": float(j["revolutions_pose"]) if j.get("revolutions_pose") is not None else None,
		"pose_meta": dict(j["pose_meta"]) if j.get("pose_meta") is not None else None,
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


async def get_jump_with_imu_by_jump_id(jump_id: int) -> Optional[Dict[str, Any]]:
	"""
	Fetch one jump row and its IMU samples by jumps.id (jump_id).

	This is the preferred method because event_id is not guaranteed unique across sessions.
	"""
	global _pool
	if _pool is None:
		return None
	jid = int(jump_id)
	async with _pool.acquire() as conn:
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
	jframes = await get_jump_frames(jid)

	return {
		"jump_id": jid,
		"event_id": j["event_id"],
		"session_id": j["session_id"],
		"video_path": j["video_path"],
		"t_peak": j["t_peak"].timestamp() if j["t_peak"] else None,
		"t_start": j["t_start"].timestamp() if j["t_start"] else None,
		"t_end": j["t_end"].timestamp() if j["t_end"] else None,
		"t_takeoff_calc": j["t_takeoff_calc"].timestamp() if j.get("t_takeoff_calc") else None,
		"t_landing_calc": j["t_landing_calc"].timestamp() if j.get("t_landing_calc") else None,
		"t_takeoff_video": j["t_takeoff_video"].timestamp() if j.get("t_takeoff_video") else None,
		"t_takeoff_video_t": float(j["t_takeoff_video_t"]) if j.get("t_takeoff_video_t") is not None else None,
		"t_landing_video": j["t_landing_video"].timestamp() if j.get("t_landing_video") else None,
		"t_landing_video_t": float(j["t_landing_video_t"]) if j.get("t_landing_video_t") is not None else None,
		"theta_z_rad": float(j["theta_z_rad"]) if j.get("theta_z_rad") is not None else None,
		"revolutions_est": float(j["revolutions_est"]) if j.get("revolutions_est") is not None else None,
		"revolutions_class": int(j["revolutions_class"]) if j.get("revolutions_class") is not None else None,
		"underrotation": float(j["underrotation"]) if j.get("underrotation") is not None else None,
		"underrot_flag": bool(j["underrot_flag"]) if j.get("underrot_flag") is not None else None,
		"gz_bias": float(j["gz_bias"]) if j.get("gz_bias") is not None else None,
		"flight_time_marked": float(j["flight_time_marked"]) if j.get("flight_time_marked") is not None else None,
		"height_marked": float(j["height_marked"]) if j.get("height_marked") is not None else None,
		"rotation_phase_marked": float(j["rotation_phase_marked"]) if j.get("rotation_phase_marked") is not None else None,
		"theta_z_rad_marked": float(j["theta_z_rad_marked"]) if j.get("theta_z_rad_marked") is not None else None,
		"revolutions_est_marked": float(j["revolutions_est_marked"]) if j.get("revolutions_est_marked") is not None else None,
		"revolutions_class_marked": int(j["revolutions_class_marked"]) if j.get("revolutions_class_marked") is not None else None,
		"underrotation_marked": float(j["underrotation_marked"]) if j.get("underrotation_marked") is not None else None,
		"underrot_flag_marked": bool(j["underrot_flag_marked"]) if j.get("underrot_flag_marked") is not None else None,
		"gz_bias_marked": float(j["gz_bias_marked"]) if j.get("gz_bias_marked") is not None else None,
		"flight_time_pose": float(j["flight_time_pose"]) if j.get("flight_time_pose") is not None else None,
		"height_pose": float(j["height_pose"]) if j.get("height_pose") is not None else None,
		"revolutions_pose": float(j["revolutions_pose"]) if j.get("revolutions_pose") is not None else None,
		"pose_meta": dict(j["pose_meta"]) if j.get("pose_meta") is not None else None,
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
	global _pool
	if _pool is None:
		return None
	jid = int(jump_id) if jump_id is not None else None
	eid = int(event_id) if event_id is not None else None
	sid = (session_id or "").strip() or None
	async with _pool.acquire() as conn:
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
	- `imu_samples.jump_id` and `jump_frames.jump_id` have `ON DELETE CASCADE`, so deleting the jump row
	  deletes its samples and frames automatically.
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
	global _pool
	if _pool is None:
		print("[DB] delete_jump_by_jump_id: _pool is None, skipping")
		return {"deleted": False, "detail": "DB not configured"}
	jid = int(jump_id)
	async with _pool.acquire() as conn:
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
	global _pool
	if _pool is None:
		print("[DB] delete_jumps_bulk: _pool is None, skipping")
		return {"deleted_count": 0, "detail": "DB not configured"}
	
	if not jump_ids:
		return {"deleted_count": 0, "detail": "No jump IDs provided"}
	
	# Convert to integers and filter out invalid values
	jids = [int(jid) for jid in jump_ids if isinstance(jid, (int, str)) and str(jid).strip()]
	if not jids:
		return {"deleted_count": 0, "detail": "No valid jump IDs provided"}
	
	async with _pool.acquire() as conn:
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
async def list_devices() -> List[Dict[str, Any]]:
	"""
	List all registered devices.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			ORDER BY name, mac_address;
			"""
		)
		return [
			{
				"id": int(r["id"]),
				"mac_address": str(r["mac_address"]),
				"name": str(r["name"]),
				"created_at": r["created_at"].isoformat() if r["created_at"] else None,
				"updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
			}
			for r in rows
		]


async def get_device_by_mac(mac_address: str) -> Optional[Dict[str, Any]]:
	"""
	Get device by MAC address.
	"""
	global _pool
	if _pool is None:
		return None
	mac = (mac_address or "").strip().upper()
	if not mac:
		return None
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			WHERE mac_address = $1;
			""",
			mac,
		)
		if not row:
			return None
		return {
			"id": int(row["id"]),
			"mac_address": str(row["mac_address"]),
			"name": str(row["name"]),
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def get_device_by_name(name: str) -> Optional[Dict[str, Any]]:
	"""
	Get device by name.
	"""
	global _pool
	if _pool is None:
		return None
	name_str = (name or "").strip()
	if not name_str:
		return None
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, mac_address, name, created_at, updated_at
			FROM devices
			WHERE name = $1;
			""",
			name_str,
		)
		if not row:
			return None
		return {
			"id": int(row["id"]),
			"mac_address": str(row["mac_address"]),
			"name": str(row["name"]),
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def upsert_device(mac_address: str, name: str) -> Dict[str, Any]:
	"""
	Create or update a device mapping. Returns the device record.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	mac = (mac_address or "").strip().upper()
	name_str = (name or "").strip()
	if not mac:
		raise ValueError("MAC address is required")
	if not name_str:
		raise ValueError("Device name is required")
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			INSERT INTO devices (mac_address, name)
			VALUES ($1, $2)
			ON CONFLICT (mac_address) DO UPDATE SET
				name = EXCLUDED.name,
				updated_at = NOW()
			RETURNING id, mac_address, name, created_at, updated_at;
			""",
			mac,
			name_str,
		)
		if not row:
			raise RuntimeError("Failed to upsert device")
		return {
			"id": int(row["id"]),
			"mac_address": str(row["mac_address"]),
			"name": str(row["name"]),
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def delete_device(device_id: int) -> Dict[str, Any]:
	"""
	Delete a device by ID.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM devices
			WHERE id = $1
			RETURNING id, mac_address, name;
			""",
			device_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No device found with id={device_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"mac_address": str(row["mac_address"]),
			"name": str(row["name"]),
		}


async def resolve_device_identifier(identifier: str) -> Optional[str]:
	"""
	Resolve a device identifier (MAC address or name) to MAC address.
	Returns the MAC address if found, None otherwise.
	"""
	global _pool
	if _pool is None:
		return None
	identifier_str = (identifier or "").strip()
	if not identifier_str:
		return None
	
	# If it looks like a MAC address, normalize and check
	if ":" in identifier_str or len(identifier_str) == 17:
		mac = identifier_str.upper()
		device = await get_device_by_mac(mac)
		if device:
			return device["mac_address"]
		# Also return the MAC as-is if it's valid format (might not be in DB yet)
		return mac
	
	# Otherwise, treat as name and look up
	device = await get_device_by_name(identifier_str)
	if device:
		return device["mac_address"]
	
	return None


# Skater management functions
async def list_skaters() -> List[Dict[str, Any]]:
	"""
	List all registered skaters.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at
			FROM skaters
			ORDER BY name;
			"""
		)
		return [
			{
				"id": int(r["id"]),
				"name": str(r["name"]),
				"date_of_birth": r["date_of_birth"].isoformat() if r["date_of_birth"] else None,
				"gender": str(r["gender"]) if r["gender"] else None,
				"level": str(r["level"]) if r["level"] else None,
				"club": str(r["club"]) if r["club"] else None,
				"email": str(r["email"]) if r["email"] else None,
				"phone": str(r["phone"]) if r["phone"] else None,
				"notes": str(r["notes"]) if r["notes"] else None,
				"created_at": r["created_at"].isoformat() if r["created_at"] else None,
				"updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
			}
			for r in rows
		]


async def get_skater_by_id(skater_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get skater by ID, including coaches and devices.
	"""
	global _pool
	if _pool is None:
		return None
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at
			FROM skaters
			WHERE id = $1;
			""",
			skater_id,
		)
		if not row:
			return None
		skater = {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"date_of_birth": row["date_of_birth"].isoformat() if row["date_of_birth"] else None,
			"gender": str(row["gender"]) if row["gender"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}
		# Include coaches and devices
		skater["coaches"] = await get_skater_coaches(skater_id)
		skater["devices"] = await get_skater_devices(skater_id)
		return skater


async def upsert_skater(
	name: str,
	date_of_birth: Optional[str] = None,
	gender: Optional[str] = None,
	level: Optional[str] = None,
	club: Optional[str] = None,
	email: Optional[str] = None,
	phone: Optional[str] = None,
	notes: Optional[str] = None,
	skater_id: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Create or update a skater profile.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	name_str = (name or "").strip()
	if not name_str:
		raise ValueError("Skater name is required")
	
	# Parse date_of_birth if provided
	dob_date = None
	if date_of_birth:
		try:
			from datetime import datetime
			dob_date = datetime.fromisoformat(date_of_birth.replace("Z", "+00:00")).date()
		except Exception:
			pass
	
	async with _pool.acquire() as conn:
		if skater_id:
			# Update existing
			row = await conn.fetchrow(
				"""
				UPDATE skaters
				SET name = $1, date_of_birth = $2, gender = $3, level = $4, club = $5,
				    email = $6, phone = $7, notes = $8, updated_at = NOW()
				WHERE id = $9
				RETURNING id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at;
				""",
				name_str,
				dob_date,
				gender.strip() if gender else None,
				level.strip() if level else None,
				club.strip() if club else None,
				email.strip() if email else None,
				phone.strip() if phone else None,
				notes.strip() if notes else None,
				skater_id,
			)
		else:
			# Create new
			row = await conn.fetchrow(
				"""
				INSERT INTO skaters (name, date_of_birth, gender, level, club, email, phone, notes)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
				RETURNING id, name, date_of_birth, gender, level, club, email, phone, notes, created_at, updated_at;
				""",
				name_str,
				dob_date,
				gender.strip() if gender else None,
				level.strip() if level else None,
				club.strip() if club else None,
				email.strip() if email else None,
				phone.strip() if phone else None,
				notes.strip() if notes else None,
			)
		if not row:
			raise RuntimeError("Failed to upsert skater")
		return {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"date_of_birth": row["date_of_birth"].isoformat() if row["date_of_birth"] else None,
			"gender": str(row["gender"]) if row["gender"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def delete_skater(skater_id: int) -> Dict[str, Any]:
	"""
	Delete a skater by ID.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM skaters
			WHERE id = $1
			RETURNING id, name;
			""",
			skater_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No skater found with id={skater_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"name": str(row["name"]),
		}


# Coach management functions
async def list_coaches() -> List[Dict[str, Any]]:
	"""
	List all registered coaches.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT id, name, email, phone, certification, level, club, notes, created_at, updated_at
			FROM coaches
			ORDER BY name;
			"""
		)
		return [
			{
				"id": int(r["id"]),
				"name": str(r["name"]),
				"email": str(r["email"]) if r["email"] else None,
				"phone": str(r["phone"]) if r["phone"] else None,
				"certification": str(r["certification"]) if r["certification"] else None,
				"level": str(r["level"]) if r["level"] else None,
				"club": str(r["club"]) if r["club"] else None,
				"notes": str(r["notes"]) if r["notes"] else None,
				"created_at": r["created_at"].isoformat() if r["created_at"] else None,
				"updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
			}
			for r in rows
		]


async def get_coach_by_id(coach_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get coach by ID, including assigned skaters.
	"""
	global _pool
	if _pool is None:
		return None
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT id, name, email, phone, certification, level, club, notes, created_at, updated_at
			FROM coaches
			WHERE id = $1;
			""",
			coach_id,
		)
		if not row:
			return None
		coach = {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"certification": str(row["certification"]) if row["certification"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}
		# Include assigned skaters
		coach["skaters"] = await get_coach_skaters(coach_id)
		return coach


async def upsert_coach(
	name: str,
	email: Optional[str] = None,
	phone: Optional[str] = None,
	certification: Optional[str] = None,
	level: Optional[str] = None,
	club: Optional[str] = None,
	notes: Optional[str] = None,
	coach_id: Optional[int] = None,
) -> Dict[str, Any]:
	"""
	Create or update a coach profile.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	name_str = (name or "").strip()
	if not name_str:
		raise ValueError("Coach name is required")
	
	async with _pool.acquire() as conn:
		if coach_id:
			# Update existing
			row = await conn.fetchrow(
				"""
				UPDATE coaches
				SET name = $1, email = $2, phone = $3, certification = $4, level = $5,
				    club = $6, notes = $7, updated_at = NOW()
				WHERE id = $8
				RETURNING id, name, email, phone, certification, level, club, notes, created_at, updated_at;
				""",
				name_str,
				email.strip() if email else None,
				phone.strip() if phone else None,
				certification.strip() if certification else None,
				level.strip() if level else None,
				club.strip() if club else None,
				notes.strip() if notes else None,
				coach_id,
			)
		else:
			# Create new
			row = await conn.fetchrow(
				"""
				INSERT INTO coaches (name, email, phone, certification, level, club, notes)
				VALUES ($1, $2, $3, $4, $5, $6, $7)
				RETURNING id, name, email, phone, certification, level, club, notes, created_at, updated_at;
				""",
				name_str,
				email.strip() if email else None,
				phone.strip() if phone else None,
				certification.strip() if certification else None,
				level.strip() if level else None,
				club.strip() if club else None,
				notes.strip() if notes else None,
			)
		if not row:
			raise RuntimeError("Failed to upsert coach")
		return {
			"id": int(row["id"]),
			"name": str(row["name"]),
			"email": str(row["email"]) if row["email"] else None,
			"phone": str(row["phone"]) if row["phone"] else None,
			"certification": str(row["certification"]) if row["certification"] else None,
			"level": str(row["level"]) if row["level"] else None,
			"club": str(row["club"]) if row["club"] else None,
			"notes": str(row["notes"]) if row["notes"] else None,
			"created_at": row["created_at"].isoformat() if row["created_at"] else None,
			"updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
		}


async def delete_coach(coach_id: int) -> Dict[str, Any]:
	"""
	Delete a coach by ID.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			DELETE FROM coaches
			WHERE id = $1
			RETURNING id, name;
			""",
			coach_id,
		)
		if not row:
			return {"deleted": False, "detail": f"No coach found with id={coach_id}"}
		return {
			"deleted": True,
			"id": int(row["id"]),
			"name": str(row["name"]),
		}


# Skater-Coach relationship functions
async def get_skater_coaches(skater_id: int) -> List[Dict[str, Any]]:
	"""
	Get all coaches assigned to a skater.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sc.id, sc.coach_id, sc.is_head_coach, c.name as coach_name
			FROM skater_coaches sc
			JOIN coaches c ON sc.coach_id = c.id
			WHERE sc.skater_id = $1
			ORDER BY sc.is_head_coach DESC, c.name;
			""",
			skater_id,
		)
		return [
			{
				"id": int(r["id"]),
				"coach_id": int(r["coach_id"]),
				"coach_name": str(r["coach_name"]),
				"is_head_coach": bool(r["is_head_coach"]),
			}
			for r in rows
		]


async def add_skater_coach(skater_id: int, coach_id: int, is_head_coach: bool = False) -> Dict[str, Any]:
	"""
	Add a coach to a skater. If is_head_coach is True, unset other head coaches for this skater.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		async with conn.transaction():
			# If setting as head coach, unset other head coaches
			if is_head_coach:
				await conn.execute(
					"UPDATE skater_coaches SET is_head_coach = FALSE WHERE skater_id = $1",
					skater_id
				)
			# Insert or update relationship
			row = await conn.fetchrow(
				"""
				INSERT INTO skater_coaches (skater_id, coach_id, is_head_coach)
				VALUES ($1, $2, $3)
				ON CONFLICT (skater_id, coach_id) DO UPDATE SET is_head_coach = EXCLUDED.is_head_coach
				RETURNING id, skater_id, coach_id, is_head_coach;
				""",
				skater_id,
				coach_id,
				is_head_coach,
			)
			if not row:
				raise RuntimeError("Failed to add skater-coach relationship")
			return {
				"id": int(row["id"]),
				"skater_id": int(row["skater_id"]),
				"coach_id": int(row["coach_id"]),
				"is_head_coach": bool(row["is_head_coach"]),
			}


async def remove_skater_coach(skater_id: int, coach_id: int) -> bool:
	"""
	Remove a coach from a skater.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		result = await conn.execute(
			"DELETE FROM skater_coaches WHERE skater_id = $1 AND coach_id = $2",
			skater_id,
			coach_id,
		)
		return result == "DELETE 1"


# Skater-Device relationship functions
async def get_skater_devices(skater_id: int) -> List[Dict[str, Any]]:
	"""
	Get all devices assigned to a skater.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sd.id, sd.device_id, sd.placement, d.name as device_name, d.mac_address
			FROM skater_devices sd
			JOIN devices d ON sd.device_id = d.id
			WHERE sd.skater_id = $1
			ORDER BY sd.placement, d.name;
			""",
			skater_id,
		)
		return [
			{
				"id": int(r["id"]),
				"device_id": int(r["device_id"]),
				"device_name": str(r["device_name"]),
				"mac_address": str(r["mac_address"]),
				"placement": str(r["placement"]),
			}
			for r in rows
		]


async def add_skater_device(skater_id: int, device_id: int, placement: str = "waist") -> Dict[str, Any]:
	"""
	Add a device to a skater with placement info.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	placement_str = (placement or "waist").strip().lower()
	if placement_str not in ["waist", "chest", "feet"]:
		placement_str = "waist"
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			INSERT INTO skater_devices (skater_id, device_id, placement)
			VALUES ($1, $2, $3)
			ON CONFLICT (skater_id, device_id) DO UPDATE SET placement = EXCLUDED.placement
			RETURNING id, skater_id, device_id, placement;
			""",
			skater_id,
			device_id,
			placement_str,
		)
		if not row:
			raise RuntimeError("Failed to add skater-device relationship")
		return {
			"id": int(row["id"]),
			"skater_id": int(row["skater_id"]),
			"device_id": int(row["device_id"]),
			"placement": str(row["placement"]),
		}


async def remove_skater_device(skater_id: int, device_id: int) -> bool:
	"""
	Remove a device from a skater.
	"""
	global _pool
	if _pool is None:
		raise RuntimeError("Database pool not initialized")
	async with _pool.acquire() as conn:
		result = await conn.execute(
			"DELETE FROM skater_devices WHERE skater_id = $1 AND device_id = $2",
			skater_id,
			device_id,
		)
		return result == "DELETE 1"


# Coach-Skater relationship functions (reverse lookup)
async def get_coach_skaters(coach_id: int) -> List[Dict[str, Any]]:
	"""
	Get all skaters assigned to a coach.
	"""
	global _pool
	if _pool is None:
		return []
	async with _pool.acquire() as conn:
		rows = await conn.fetch(
			"""
			SELECT sc.id, sc.skater_id, sc.is_head_coach, s.name as skater_name
			FROM skater_coaches sc
			JOIN skaters s ON sc.skater_id = s.id
			WHERE sc.coach_id = $1
			ORDER BY sc.is_head_coach DESC, s.name;
			""",
			coach_id,
		)
		return [
			{
				"id": int(r["id"]),
				"skater_id": int(r["skater_id"]),
				"skater_name": str(r["skater_name"]),
				"is_head_coach": bool(r["is_head_coach"]),
			}
			for r in rows
		]


# Skater detection settings functions
async def get_skater_detection_settings(skater_id: int) -> Optional[Dict[str, Any]]:
	"""
	Get detection settings for a skater. Returns None if not set (use defaults).
	"""
	global _pool
	if _pool is None:
		return None
	async with _pool.acquire() as conn:
		row = await conn.fetchrow(
			"""
			SELECT min_jump_height_m, min_jump_peak_az_no_g, min_jump_peak_gz_deg_s,
			       min_new_event_separation_s, min_revs, analysis_interval_s
			FROM skater_detection_settings
			WHERE skater_id = $1;
			""",
			skater_id,
		)
		if not row:
			return None
		return {
			"min_jump_height_m": float(row["min_jump_height_m"]) if row["min_jump_height_m"] is not None else None,
			"min_jump_peak_az_no_g": float(row["min_jump_peak_az_no_g"]) if row["min_jump_peak_az_no_g"] is not None else None,
			"min_jump_peak_gz_deg_s": float(row["min_jump_peak_gz_deg_s"]) if row["min_jump_peak_gz_deg_s"] is not None else None,
			"min_new_event_separation_s": float(row["min_new_event_separation_s"]) if row["min_new_event_separation_s"] is not None else None,
			"min_revs": float(row["min_revs"]) if row["min_revs"] is not None else None,
			"analysis_interval_s": float(row["analysis_interval_s"]) if row["analysis_interval_s"] is not None else None,
		}


async def close_db() -> None:
	"""Close the connection pool on shutdown."""
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None


