"""Connection pool, init_db, get_status, close_db, and _to_dt for the db package."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

from modules.config import get_config

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None
_last_init_error: Optional[str] = None
_init_lock = asyncio.Lock()
_warned_no_dsn: bool = False


def get_pool() -> Optional[asyncpg.Pool]:
	"""Return the connection pool (None if not initialized). Used by domain modules."""
	return _pool


def _to_dt(sec: float) -> datetime:
	"""Convert a wall‑clock seconds float to timezone‑aware datetime."""
	return datetime.fromtimestamp(sec, tz=timezone.utc)


async def init_db() -> None:
	"""
	Initialise PostgreSQL connection pool and ensure tables exist.
	If database.url is not set in config.json, this becomes a no‑op.
	"""
	global _pool, _warned_no_dsn, _last_init_error
	dsn = (get_config().database.url or "").strip()
	if not dsn:
		if not _warned_no_dsn:
			logger.warning("[DB] database.url not set in config.json; persistence disabled.")
			_warned_no_dsn = True
		_last_init_error = "database.url not set"
		return

	async with _init_lock:
		if _pool is None:
			try:
				cfg = get_config()
				_pool = await asyncpg.create_pool(
					dsn,
					min_size=max(1, int(cfg.database.pool_min_size)),
					max_size=max(max(1, int(cfg.database.pool_min_size)), int(cfg.database.pool_max_size)),
				)
				_last_init_error = None
			except Exception as e:
				_last_init_error = repr(e)
				raise

	async with _pool.acquire() as conn:
		await _create_tables(conn)


async def _create_tables(conn: asyncpg.Connection) -> None:
	"""Run all CREATE TABLE and ALTER TABLE statements. Called from init_db."""
	# Jumps table
	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS jumps (
			id            SERIAL PRIMARY KEY,
			event_id      INTEGER,
			session_id    TEXT,
			t_peak        TIMESTAMPTZ NOT NULL,
			t_start       TIMESTAMPTZ,
			t_end         TIMESTAMPTZ,
			t_takeoff_calc   TIMESTAMPTZ,
			t_landing_calc   TIMESTAMPTZ,
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
	for col in (
		"session_id TEXT", "video_path TEXT", "t_takeoff_calc TIMESTAMPTZ", "t_landing_calc TIMESTAMPTZ",
		"t_takeoff_video TIMESTAMPTZ", "t_landing_video TIMESTAMPTZ",
		"t_takeoff_video_t DOUBLE PRECISION", "t_landing_video_t DOUBLE PRECISION",
		"theta_z_rad DOUBLE PRECISION", "revolutions_est DOUBLE PRECISION", "revolutions_class INTEGER",
		"underrotation DOUBLE PRECISION", "underrot_flag BOOLEAN", "gz_bias DOUBLE PRECISION",
		"flight_time_marked DOUBLE PRECISION", "height_marked DOUBLE PRECISION",
		"rotation_phase_marked DOUBLE PRECISION", "theta_z_rad_marked DOUBLE PRECISION",
		"revolutions_est_marked DOUBLE PRECISION", "revolutions_class_marked INTEGER",
		"underrotation_marked DOUBLE PRECISION", "underrot_flag_marked BOOLEAN", "gz_bias_marked DOUBLE PRECISION",
		"flight_time_pose DOUBLE PRECISION", "height_pose DOUBLE PRECISION",
		"revolutions_pose DOUBLE PRECISION", "pose_meta JSONB",
	):
		await conn.execute(f"ALTER TABLE jumps ADD COLUMN IF NOT EXISTS {col};")

	try:
		await conn.execute(
			"""
			WITH ranked AS (
				SELECT id, ROW_NUMBER() OVER (
					PARTITION BY session_id, event_id
					ORDER BY (name IS NOT NULL AND name <> '') DESC, (note IS NOT NULL AND note <> '') DESC,
						(t_takeoff_video IS NOT NULL) DESC, (t_landing_video IS NOT NULL) DESC,
						(video_path IS NOT NULL AND video_path <> '') DESC, created_at DESC
				) AS rn
				FROM jumps WHERE session_id IS NOT NULL AND event_id IS NOT NULL
			)
			DELETE FROM jumps WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
			"""
		)
	except Exception as e:
		logger.warning("[DB] dedupe jumps failed: %s", e)
	try:
		await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS jumps_session_event_unique ON jumps(session_id, event_id);")
	except Exception as e:
		logger.warning("[DB] failed to create UNIQUE index jumps_session_event_unique: %s", e)
	await conn.execute("CREATE INDEX IF NOT EXISTS jumps_session_tpeak_idx ON jumps(session_id, t_peak DESC);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS imu_samples (
			id BIGSERIAL PRIMARY KEY,
			jump_id INTEGER REFERENCES jumps(id) ON DELETE CASCADE,
			t TIMESTAMPTZ NOT NULL,
			imu_timestamp BIGINT,
			acc_x DOUBLE PRECISION, acc_y DOUBLE PRECISION, acc_z DOUBLE PRECISION,
			gyro_x DOUBLE PRECISION, gyro_y DOUBLE PRECISION, gyro_z DOUBLE PRECISION,
			mag_x DOUBLE PRECISION, mag_y DOUBLE PRECISION, mag_z DOUBLE PRECISION
		);
		"""
	)

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS sessions (
			session_id TEXT PRIMARY KEY,
			t_start TIMESTAMPTZ, t_stop TIMESTAMPTZ,
			imu_mode TEXT, imu_rate INTEGER, video_backend TEXT, video_fps INTEGER, video_path TEXT,
			camera_clock_offset_s DOUBLE PRECISION, jump_config JSONB, meta JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
		"""
	)
	await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS video_backend TEXT;")
	await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS camera_clock_offset_s DOUBLE PRECISION;")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS frames (
			id BIGSERIAL PRIMARY KEY,
			session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
			frame_idx INTEGER NOT NULL, t_host DOUBLE PRECISION NOT NULL, device_ts DOUBLE PRECISION,
			source TEXT, width INTEGER, height INTEGER
		);
		"""
	)
	await conn.execute("ALTER TABLE frames ADD COLUMN IF NOT EXISTS source TEXT;")
	await conn.execute("CREATE INDEX IF NOT EXISTS frames_session_idx ON frames(session_id, frame_idx);")
	await conn.execute("CREATE INDEX IF NOT EXISTS frames_session_thost ON frames(session_id, t_host);")
	await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS frames_session_frame_unique ON frames(session_id, frame_idx);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS jump_frames (
			id BIGSERIAL PRIMARY KEY,
			jump_id INTEGER REFERENCES jumps(id) ON DELETE CASCADE,
			frame_idx INTEGER NOT NULL, t_video DOUBLE PRECISION NOT NULL, t_host DOUBLE PRECISION NOT NULL,
			device_ts DOUBLE PRECISION, width INTEGER, height INTEGER
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS jump_frames_jump_idx ON jump_frames(jump_id, frame_idx);")
	await conn.execute("CREATE INDEX IF NOT EXISTS jump_frames_jump_tvideo ON jump_frames(jump_id, t_video);")
	await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS jump_frames_jump_frame_unique ON jump_frames(jump_id, frame_idx);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS devices (
			id BIGSERIAL PRIMARY KEY,
			mac_address TEXT NOT NULL UNIQUE,
			name TEXT NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS devices_mac_idx ON devices(mac_address);")
	await conn.execute("CREATE INDEX IF NOT EXISTS devices_name_idx ON devices(name);")
	await conn.execute("""
		CREATE OR REPLACE FUNCTION update_devices_updated_at()
		RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
		$$ LANGUAGE plpgsql;
	""")
	await conn.execute("""
		DROP TRIGGER IF EXISTS devices_updated_at_trigger ON devices;
		CREATE TRIGGER devices_updated_at_trigger BEFORE UPDATE ON devices
		FOR EACH ROW EXECUTE FUNCTION update_devices_updated_at();
	""")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS skaters (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL, date_of_birth DATE, gender TEXT, level TEXT, club TEXT,
			email TEXT, phone TEXT, notes TEXT,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
		"""
	)
	await conn.execute("ALTER TABLE skaters ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE;")
	await conn.execute("CREATE INDEX IF NOT EXISTS skaters_name_idx ON skaters(name);")
	await conn.execute("CREATE INDEX IF NOT EXISTS skaters_level_idx ON skaters(level);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS coaches (
			id BIGSERIAL PRIMARY KEY,
			name TEXT NOT NULL, email TEXT, phone TEXT, certification TEXT, level TEXT, club TEXT, notes TEXT,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS coaches_name_idx ON coaches(name);")
	await conn.execute("CREATE INDEX IF NOT EXISTS coaches_level_idx ON coaches(level);")

	await conn.execute("""
		DROP TRIGGER IF EXISTS skaters_updated_at_trigger ON skaters;
		CREATE TRIGGER skaters_updated_at_trigger BEFORE UPDATE ON skaters
		FOR EACH ROW EXECUTE FUNCTION update_devices_updated_at();
	""")
	await conn.execute("""
		DROP TRIGGER IF EXISTS coaches_updated_at_trigger ON coaches;
		CREATE TRIGGER coaches_updated_at_trigger BEFORE UPDATE ON coaches
		FOR EACH ROW EXECUTE FUNCTION update_devices_updated_at();
	""")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS skater_coaches (
			id BIGSERIAL PRIMARY KEY,
			skater_id INTEGER NOT NULL REFERENCES skaters(id) ON DELETE CASCADE,
			coach_id INTEGER NOT NULL REFERENCES coaches(id) ON DELETE CASCADE,
			is_head_coach BOOLEAN NOT NULL DEFAULT FALSE,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			UNIQUE(skater_id, coach_id)
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS skater_coaches_skater_idx ON skater_coaches(skater_id);")
	await conn.execute("CREATE INDEX IF NOT EXISTS skater_coaches_coach_idx ON skater_coaches(coach_id);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS skater_devices (
			id BIGSERIAL PRIMARY KEY,
			skater_id INTEGER NOT NULL REFERENCES skaters(id) ON DELETE CASCADE,
			device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
			placement TEXT NOT NULL DEFAULT 'waist',
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			UNIQUE(skater_id, device_id)
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS skater_devices_skater_idx ON skater_devices(skater_id);")
	await conn.execute("CREATE INDEX IF NOT EXISTS skater_devices_device_idx ON skater_devices(device_id);")

	await conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS skater_detection_settings (
			id BIGSERIAL PRIMARY KEY,
			skater_id INTEGER NOT NULL UNIQUE REFERENCES skaters(id) ON DELETE CASCADE,
			min_jump_height_m REAL, min_jump_peak_az_no_g REAL, min_jump_peak_gz_deg_s REAL,
			min_new_event_separation_s REAL, min_revs REAL, analysis_interval_s REAL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		);
		"""
	)
	await conn.execute("CREATE INDEX IF NOT EXISTS skater_detection_settings_skater_idx ON skater_detection_settings(skater_id);")


def get_status() -> Dict[str, Any]:
	"""Return lightweight DB status for diagnostics."""
	dsn = (get_config().database.url or "").strip()
	return {
		"enabled": bool(dsn),
		"dsn_set": bool(dsn),
		"pool_ready": _pool is not None,
		"last_init_error": _last_init_error,
	}


async def close_db() -> None:
	"""Close the connection pool on shutdown."""
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None
