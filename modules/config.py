from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DatabaseConfig:
	# If empty, DB persistence is disabled (best-effort mode).
	url: str = ""


@dataclass(frozen=True)
class MovesenseConfig:
	# Used as UI defaults only; users can still connect to any device via the UI.
	default_device: str = ""
	default_mode: str = "IMU9"
	default_rate: int = 104
	ble_adapter: str = ""  # Linux: "hci0" / "hci1" etc


@dataclass(frozen=True)
class ImuUdpConfig:
	host: str = "127.0.0.1"
	port: int = 9999


@dataclass(frozen=True)
class AutoConnectConfig:
	# Seconds between IMU connection retries when default skater has device but IMU is not yet connected.
	imu_retry_interval_seconds: float = 15.0


@dataclass(frozen=True)
class JobsConfig:
	jump_clip_jobs_dir: str = str(Path("data") / "jobs" / "jump_clips")


@dataclass(frozen=True)
class SessionsConfig:
	# Base directory for session folders.
	# Expected layout:
	#   <base_dir>/<session_id>/
	#     video.mp4
	#     frames.csv
	#     jump_clips/...
	base_dir: str = str(Path("data") / "sessions")
	# Subdirectory under each session directory for per-jump clips.
	jump_clips_subdir: str = "jump_clips"


@dataclass(frozen=True)
class JumpRecordingConfig:
	# Time window around detected jump peak for saving IMU samples and cutting video clips.
	# Values are in seconds. Window is computed as [t_peak - pre_jump_seconds, t_peak + post_jump_seconds].
	pre_jump_seconds: float = 2.0  # How many seconds before t_peak to include
	post_jump_seconds: float = 3.0  # How many seconds after t_peak to include
	# Additional buffer beyond the IMU window when cutting video clips (to ensure smooth transitions).
	clip_buffer_seconds: float = 0.8  # Extra seconds added to both sides of window for clip cutting


@dataclass(frozen=True)
class Picamera2CameraConfig:
	# Optional; if omitted, backend defaults apply.
	record_size: Optional[list[int]] = None  # [w,h]
	preview_size: Optional[list[int]] = None  # [w,h]
	preview_fps: Optional[int] = None
	bitrate: Optional[int] = None
	controls: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Picamera2Config:
	primary_index: int = 1
	gs_index: int = 0
	module3: Picamera2CameraConfig = field(default_factory=Picamera2CameraConfig)
	gs: Picamera2CameraConfig = field(default_factory=Picamera2CameraConfig)


@dataclass(frozen=True)
class VideoProcessConfig:
	enabled: bool = False
	collector_host: str = "127.0.0.1"
	collector_port: int = 18081


@dataclass(frozen=True)
class VideoConfig:
	backend: str = "picamera2"  # picamera2 / jetson
	process: VideoProcessConfig = field(default_factory=VideoProcessConfig)
	picamera2: Picamera2Config = field(default_factory=Picamera2Config)


@dataclass(frozen=True)
class AppConfig:
	database: DatabaseConfig = field(default_factory=DatabaseConfig)
	movesense: MovesenseConfig = field(default_factory=MovesenseConfig)
	imu_udp: ImuUdpConfig = field(default_factory=ImuUdpConfig)
	auto_connect: AutoConnectConfig = field(default_factory=AutoConnectConfig)
	jobs: JobsConfig = field(default_factory=JobsConfig)
	sessions: SessionsConfig = field(default_factory=SessionsConfig)
	video: VideoConfig = field(default_factory=VideoConfig)
	jump_recording: JumpRecordingConfig = field(default_factory=JumpRecordingConfig)


_CONFIG_PATH: Optional[Path] = None
_CONFIG_CACHE: Optional[AppConfig] = None


def _repo_root() -> Path:
	# modules/config.py -> repo root is one level up.
	return Path(__file__).resolve().parents[1]


def get_default_config_path() -> Path:
	return _repo_root() / "config.json"


def set_config_path(path: str | Path) -> None:
	"""
	Override the config path (must be called before first get_config()).
	Intended for tooling/subprocess use; server normally uses the default path.
	"""
	global _CONFIG_PATH
	global _CONFIG_CACHE
	_CONFIG_PATH = Path(path).expanduser().resolve()
	_CONFIG_CACHE = None


def _deep_get(d: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
	cur: Any = d
	for k in keys:
		if not isinstance(cur, dict):
			return default
		cur = cur.get(k)
	return cur if cur is not None else default


def _as_int(v: Any, default: int) -> int:
	try:
		return int(v)
	except Exception:
		return int(default)


def _as_bool(v: Any, default: bool) -> bool:
	if isinstance(v, bool):
		return v
	if isinstance(v, (int, float)):
		return bool(v)
	if isinstance(v, str):
		return v.strip().lower() in ("1", "true", "yes", "on")
	return bool(default)


def _as_str(v: Any, default: str = "") -> str:
	return str(v) if v is not None else str(default)


def _as_float(v: Any, default: float) -> float:
	try:
		return float(v)
	except Exception:
		return float(default)


def _parse_camera_cfg(obj: Any) -> Picamera2CameraConfig:
	if not isinstance(obj, dict):
		return Picamera2CameraConfig()
	controls = obj.get("controls")
	if controls is None or not isinstance(controls, dict):
		controls_dict: Dict[str, Any] = {}
	else:
		controls_dict = dict(controls)
	return Picamera2CameraConfig(
		record_size=obj.get("record_size") if isinstance(obj.get("record_size"), list) else None,
		preview_size=obj.get("preview_size") if isinstance(obj.get("preview_size"), list) else None,
		preview_fps=_as_int(obj.get("preview_fps"), None) if obj.get("preview_fps") is not None else None,
		bitrate=_as_int(obj.get("bitrate"), None) if obj.get("bitrate") is not None else None,
		controls=controls_dict,
	)


def load_config(path: Optional[str | Path] = None) -> AppConfig:
	p = Path(path).expanduser().resolve() if path else (_CONFIG_PATH or get_default_config_path())
	if not p.exists():
		# Defaults-only config; app can still run.
		return AppConfig()
	try:
		raw = json.loads(p.read_text(encoding="utf-8"))
	except Exception:
		# If config is malformed, fail safe to defaults (but keep app running).
		return AppConfig()

	if not isinstance(raw, dict):
		return AppConfig()

	db_url = _as_str(_deep_get(raw, ["database", "url"], ""), "")

	m_default_device = _as_str(_deep_get(raw, ["movesense", "default_device"], ""), "")
	m_default_mode = _as_str(_deep_get(raw, ["movesense", "default_mode"], "IMU9"), "IMU9")
	m_default_rate = _as_int(_deep_get(raw, ["movesense", "default_rate"], 104), 104)
	m_ble_adapter = _as_str(_deep_get(raw, ["movesense", "ble_adapter"], ""), "")

	udp_host = _as_str(_deep_get(raw, ["imu_udp", "host"], "127.0.0.1"), "127.0.0.1")
	udp_port = _as_int(_deep_get(raw, ["imu_udp", "port"], 9999), 9999)

	imu_retry_sec = _as_float(_deep_get(raw, ["auto_connect", "imu_retry_interval_seconds"], 15.0), 15.0)
	imu_retry_sec = max(1.0, float(imu_retry_sec))

	jobs_dir = _as_str(_deep_get(raw, ["jobs", "jump_clip_jobs_dir"], str(Path("data") / "jobs" / "jump_clips")), "")

	sessions_base_dir = _as_str(_deep_get(raw, ["sessions", "base_dir"], str(Path("data") / "sessions")), "")
	jump_clips_subdir = _as_str(_deep_get(raw, ["sessions", "jump_clips_subdir"], "jump_clips"), "jump_clips").strip()
	if not jump_clips_subdir:
		jump_clips_subdir = "jump_clips"

	jump_rec_pre = _as_float(_deep_get(raw, ["jump_recording", "pre_jump_seconds"], 2.0), 2.0)
	jump_rec_post = _as_float(_deep_get(raw, ["jump_recording", "post_jump_seconds"], 3.0), 3.0)
	jump_rec_buffer = _as_float(_deep_get(raw, ["jump_recording", "clip_buffer_seconds"], 0.8), 0.8)

	video_backend = _as_str(_deep_get(raw, ["video", "backend"], "picamera2"), "picamera2").strip().lower()
	proc_enabled = _as_bool(_deep_get(raw, ["video", "process", "enabled"], False), False)
	proc_host = _as_str(_deep_get(raw, ["video", "process", "collector_host"], "127.0.0.1"), "127.0.0.1")
	proc_port = _as_int(_deep_get(raw, ["video", "process", "collector_port"], 18081), 18081)

	pc2_primary_idx = _as_int(_deep_get(raw, ["video", "picamera2", "primary_index"], 1), 1)
	pc2_gs_idx = _as_int(_deep_get(raw, ["video", "picamera2", "gs_index"], 0), 0)
	pc2_module3 = _parse_camera_cfg(_deep_get(raw, ["video", "picamera2", "module3"], {}))
	pc2_gs = _parse_camera_cfg(_deep_get(raw, ["video", "picamera2", "gs"], {}))

	return AppConfig(
		database=DatabaseConfig(url=db_url),
		movesense=MovesenseConfig(
			default_device=m_default_device,
			default_mode=m_default_mode,
			default_rate=int(m_default_rate) if int(m_default_rate) > 0 else 104,
			ble_adapter=m_ble_adapter,
		),
		imu_udp=ImuUdpConfig(host=udp_host, port=int(udp_port) if int(udp_port) > 0 else 9999),
		auto_connect=AutoConnectConfig(imu_retry_interval_seconds=float(imu_retry_sec)),
		jobs=JobsConfig(jump_clip_jobs_dir=jobs_dir),
		sessions=SessionsConfig(base_dir=sessions_base_dir, jump_clips_subdir=jump_clips_subdir),
		video=VideoConfig(
			backend=video_backend or "picamera2",
			process=VideoProcessConfig(enabled=proc_enabled, collector_host=proc_host, collector_port=int(proc_port)),
			picamera2=Picamera2Config(
				primary_index=int(pc2_primary_idx),
				gs_index=int(pc2_gs_idx),
				module3=pc2_module3,
				gs=pc2_gs,
			),
		),
		jump_recording=JumpRecordingConfig(
			pre_jump_seconds=float(jump_rec_pre) if float(jump_rec_pre) > 0.0 else 2.0,
			post_jump_seconds=float(jump_rec_post) if float(jump_rec_post) > 0.0 else 3.0,
			clip_buffer_seconds=float(jump_rec_buffer) if float(jump_rec_buffer) >= 0.0 else 0.8,
		),
	)


def get_config() -> AppConfig:
	global _CONFIG_CACHE
	if _CONFIG_CACHE is None:
		_CONFIG_CACHE = load_config()
	return _CONFIG_CACHE


