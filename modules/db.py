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


def _to_dt(sec: float) -> datetime:
	"""Convert a wall‑clock seconds float to timezone‑aware datetime."""
	return datetime.fromtimestamp(sec, tz=timezone.utc)


async def insert_jump_with_imu(
	jump: Dict[str, Any],
	annotation: Dict[str, Any],
	imu_samples: Sequence[Dict[str, Any]],
) -> None:
	"""
	Insert one jump row and its associated IMU samples into the database.

	- jump: dict containing event_id, t_peak, t_start, t_end, and metrics.
	- annotation: dict with optional 'name', 'note'.
	- imu_samples: list of rows from _imu_history (t, imu_timestamp, acc, gyro, mag).
	"""
	global _pool
	if _pool is None:
		print("[DB] insert_jump_with_imu: _pool is None, skipping")
		return
	print(f"[DB] insert_jump_with_imu: event_id={jump.get('event_id')}, samples={len(imu_samples)}")

	event_id = int(jump.get("event_id", 0)) or None
	t_peak = float(jump.get("t_peak", 0.0))
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
				event_id, t_peak, t_start, t_end,
				flight_time, height, acc_peak, gyro_peak,
				rotation_phase, confidence, name, note
			)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
			RETURNING id;
			""",
			event_id,
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
			return

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


async def close_db() -> None:
	"""Close the connection pool on shutdown."""
	global _pool
	if _pool is not None:
		await _pool.close()
		_pool = None


