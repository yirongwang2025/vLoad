from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DatabaseConfig:
	# If empty, DB persistence is disabled (best-effort mode).
	url: str = ""
	pool_min_size: int = 1
	pool_max_size: int = 5


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
	# When True, auto-enable jump detection once auto-connect succeeds.
	jump_detection_enabled: bool = True


@dataclass(frozen=True)
class BuffersConfig:
	# In-memory retention windows.
	imu_history_seconds: float = 60.0
	frame_history_seconds: float = 120.0
	# Queue/deque capacities.
	jump_sample_queue_maxsize: int = 2000
	jump_events_maxlen: int = 1000


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
class JumpDetectionConfig:
	# Detector rolling window and event-candidate timing.
	window_seconds: float = 3.0
	min_peak_height_m_s2: float = 3.0
	min_peak_distance_s: float = 0.18
	min_flight_time_s: float = 0.25
	max_flight_time_s: float = 0.80
	# Event selection thresholds.
	min_jump_height_m: float = 0.15
	min_jump_peak_az_no_g: float = 3.5
	min_jump_peak_gz_deg_s: float = 180.0
	min_new_event_separation_s: float = 0.5
	# Temporal spacing guards for event geometry (seconds).
	min_takeoff_to_peak_s: float = 0.02
	max_takeoff_to_peak_s: float = 0.80
	min_peak_to_landing_s: float = 0.02
	max_peak_to_landing_s: float = 0.80
	min_revs: float = 0.0
	analysis_interval_s: float = 0.5
	smoothing_cutoff_hz: float = 10.0
	refine_window_s: float = 0.25
	bias_window_start_s: float = 0.5
	bias_window_end_s: float = 0.1


@dataclass(frozen=True)
class PoseConfig:
	# Runtime pose extraction defaults.
	max_fps: float = 10.0
	model_complexity: int = 1
	min_detection_confidence: float = 0.5
	min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class ExportConfig:
	default_seconds: float = 30.0
	jump_fallback_half_window_seconds: float = 0.75


@dataclass(frozen=True)
class ClipWorkerConfig:
	poll_seconds: float = 0.5


@dataclass(frozen=True)
class CollectorConfig:
	# BLE/collector resilience tuning.
	payload_queue_maxsize: int = 600
	calib_warmup_seconds: float = 1.0
	calib_min_samples: int = 6
	offset_alpha: float = 0.02
	initial_backoff_seconds: float = 0.5
	max_backoff_seconds: float = 8.0
	backoff_multiplier: float = 1.6
	connect_timeout_seconds: float = 25.0
	notify_stale_timeout_seconds: float = 10.0
	stat_emit_interval_seconds: float = 1.0
	subscribe_ref_id: int = 99
	hello_ref_id: int = 1


@dataclass(frozen=True)
class BleConfig:
	# Device discovery/connection timing.
	scan_timeout_seconds: float = 7.0
	connect_find_timeout_seconds: float = 10.0
	connect_discovery_timeout_seconds: float = 7.0
	connect_fallback_scan_timeout_seconds: float = 7.0
	connect_refind_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class ApiConfig:
	# REST/view defaults.
	jumps_list_limit_default: int = 200
	jumps_list_limit_max: int = 1000
	pages_jumps_preload_limit: int = 200
	session_frames_limit_default: int = 200000


@dataclass(frozen=True)
class HttpProxyConfig:
	status_timeout_seconds: float = 2.0
	post_timeout_seconds: float = 5.0
	mjpeg_open_timeout_seconds: float = 10.0
	snapshot_timeout_seconds: float = 3.0
	mjpeg_read_chunk_bytes: int = 8192


@dataclass(frozen=True)
class RuntimeConfig:
	# Window overlap handling for persisted jump windows.
	overlap_tolerance_seconds: float = 0.1
	min_window_size_seconds: float = 2.0
	fallback_window_half_seconds: float = 1.0
	# Waiting for future IMU samples before persisting.
	imu_wait_poll_seconds: float = 0.02
	imu_wait_slack_seconds: float = 0.5
	# Limits used by maintenance/backfill code paths.
	backfill_jumps_limit: int = 2000
	backfill_frames_limit: int = 500000
	# Loop and wait timing.
	jump_worker_idle_sleep_seconds: float = 0.1
	frame_sync_interval_seconds: float = 0.1
	frame_sync_error_backoff_seconds: float = 0.5
	video_start_wait_seconds: float = 3.0
	video_status_poll_seconds: float = 0.1
	video_recording_poll_delay_seconds: float = 0.05
	video_connect_warmup_seconds: float = 0.2
	session_video_wait_timeout_seconds: float = 15.0
	session_video_wait_poll_seconds: float = 0.25
	# Jump window tweaks.
	takeoff_landing_edge_guard_seconds: float = 0.5
	clip_fallback_extra_seconds: float = 0.4
	clip_wait_mp4_timeout_seconds: float = 900.0
	jump_windows_history_per_session: int = 100
	imu_rx_rate_window_seconds: float = 5.0
	imu_history_prune_check_min_len: int = 100
	subprocess_wait_timeout_seconds: float = 3.0
	picamera_thread_join_timeout_seconds: float = 2.0
	picamera_run_loop_sleep_seconds: float = 0.05


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
	# API/runtime defaults
	default_mjpeg_fps: float = 15.0
	recording_fps: int = 30
	preview_jpeg_quality: int = 80


@dataclass(frozen=True)
class AppConfig:
	database: DatabaseConfig = field(default_factory=DatabaseConfig)
	movesense: MovesenseConfig = field(default_factory=MovesenseConfig)
	imu_udp: ImuUdpConfig = field(default_factory=ImuUdpConfig)
	auto_connect: AutoConnectConfig = field(default_factory=AutoConnectConfig)
	buffers: BuffersConfig = field(default_factory=BuffersConfig)
	jobs: JobsConfig = field(default_factory=JobsConfig)
	sessions: SessionsConfig = field(default_factory=SessionsConfig)
	video: VideoConfig = field(default_factory=VideoConfig)
	jump_recording: JumpRecordingConfig = field(default_factory=JumpRecordingConfig)
	jump_detection: JumpDetectionConfig = field(default_factory=JumpDetectionConfig)
	pose: PoseConfig = field(default_factory=PoseConfig)
	export: ExportConfig = field(default_factory=ExportConfig)
	clip_worker: ClipWorkerConfig = field(default_factory=ClipWorkerConfig)
	collector: CollectorConfig = field(default_factory=CollectorConfig)
	ble: BleConfig = field(default_factory=BleConfig)
	api: ApiConfig = field(default_factory=ApiConfig)
	http_proxy: HttpProxyConfig = field(default_factory=HttpProxyConfig)
	runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


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
	db_pool_min = _as_int(_deep_get(raw, ["database", "pool_min_size"], 1), 1)
	db_pool_max = _as_int(_deep_get(raw, ["database", "pool_max_size"], 5), 5)

	m_default_device = _as_str(_deep_get(raw, ["movesense", "default_device"], ""), "")
	m_default_mode = _as_str(_deep_get(raw, ["movesense", "default_mode"], "IMU9"), "IMU9")
	m_default_rate = _as_int(_deep_get(raw, ["movesense", "default_rate"], 104), 104)
	m_ble_adapter = _as_str(_deep_get(raw, ["movesense", "ble_adapter"], ""), "")

	udp_host = _as_str(_deep_get(raw, ["imu_udp", "host"], "127.0.0.1"), "127.0.0.1")
	udp_port = _as_int(_deep_get(raw, ["imu_udp", "port"], 9999), 9999)

	imu_retry_sec = _as_float(_deep_get(raw, ["auto_connect", "imu_retry_interval_seconds"], 15.0), 15.0)
	imu_retry_sec = max(1.0, float(imu_retry_sec))
	auto_jump_detection = _as_bool(_deep_get(raw, ["auto_connect", "jump_detection_enabled"], True), True)

	jobs_dir = _as_str(_deep_get(raw, ["jobs", "jump_clip_jobs_dir"], str(Path("data") / "jobs" / "jump_clips")), "")
	if not jobs_dir.strip():
		jobs_dir = str(Path("data") / "jobs" / "jump_clips")

	imu_hist_s = _as_float(_deep_get(raw, ["buffers", "imu_history_seconds"], 60.0), 60.0)
	frame_hist_s = _as_float(_deep_get(raw, ["buffers", "frame_history_seconds"], 120.0), 120.0)
	jump_q_max = _as_int(_deep_get(raw, ["buffers", "jump_sample_queue_maxsize"], 2000), 2000)
	jump_events_maxlen = _as_int(_deep_get(raw, ["buffers", "jump_events_maxlen"], 1000), 1000)

	sessions_base_dir = _as_str(_deep_get(raw, ["sessions", "base_dir"], str(Path("data") / "sessions")), "")
	if not sessions_base_dir.strip():
		sessions_base_dir = str(Path("data") / "sessions")
	jump_clips_subdir = _as_str(_deep_get(raw, ["sessions", "jump_clips_subdir"], "jump_clips"), "jump_clips").strip()
	if not jump_clips_subdir:
		jump_clips_subdir = "jump_clips"

	jump_rec_pre = _as_float(_deep_get(raw, ["jump_recording", "pre_jump_seconds"], 2.0), 2.0)
	jump_rec_post = _as_float(_deep_get(raw, ["jump_recording", "post_jump_seconds"], 3.0), 3.0)
	jump_rec_buffer = _as_float(_deep_get(raw, ["jump_recording", "clip_buffer_seconds"], 0.8), 0.8)

	jd_window = _as_float(_deep_get(raw, ["jump_detection", "window_seconds"], 3.0), 3.0)
	jd_min_peak = _as_float(_deep_get(raw, ["jump_detection", "min_peak_height_m_s2"], 3.0), 3.0)
	jd_peak_dist = _as_float(_deep_get(raw, ["jump_detection", "min_peak_distance_s"], 0.18), 0.18)
	jd_min_flight = _as_float(_deep_get(raw, ["jump_detection", "min_flight_time_s"], 0.25), 0.25)
	jd_max_flight = _as_float(_deep_get(raw, ["jump_detection", "max_flight_time_s"], 0.80), 0.80)
	jd_min_jump_h = _as_float(_deep_get(raw, ["jump_detection", "min_jump_height_m"], 0.15), 0.15)
	jd_min_az = _as_float(_deep_get(raw, ["jump_detection", "min_jump_peak_az_no_g"], 3.5), 3.5)
	jd_min_gz = _as_float(_deep_get(raw, ["jump_detection", "min_jump_peak_gz_deg_s"], 180.0), 180.0)
	jd_min_sep = _as_float(_deep_get(raw, ["jump_detection", "min_new_event_separation_s"], 0.5), 0.5)
	jd_min_to_peak = _as_float(_deep_get(raw, ["jump_detection", "min_takeoff_to_peak_s"], 0.02), 0.02)
	jd_max_to_peak = _as_float(_deep_get(raw, ["jump_detection", "max_takeoff_to_peak_s"], 0.80), 0.80)
	jd_min_peak_to_land = _as_float(_deep_get(raw, ["jump_detection", "min_peak_to_landing_s"], 0.02), 0.02)
	jd_max_peak_to_land = _as_float(_deep_get(raw, ["jump_detection", "max_peak_to_landing_s"], 0.80), 0.80)
	jd_min_revs = _as_float(_deep_get(raw, ["jump_detection", "min_revs"], 0.0), 0.0)
	jd_interval = _as_float(_deep_get(raw, ["jump_detection", "analysis_interval_s"], 0.5), 0.5)
	jd_smoothing_cutoff = _as_float(_deep_get(raw, ["jump_detection", "smoothing_cutoff_hz"], 10.0), 10.0)
	jd_refine_window = _as_float(_deep_get(raw, ["jump_detection", "refine_window_s"], 0.25), 0.25)
	jd_bias_start = _as_float(_deep_get(raw, ["jump_detection", "bias_window_start_s"], 0.5), 0.5)
	jd_bias_end = _as_float(_deep_get(raw, ["jump_detection", "bias_window_end_s"], 0.1), 0.1)

	pose_max_fps = _as_float(_deep_get(raw, ["pose", "max_fps"], 10.0), 10.0)
	pose_model = _as_int(_deep_get(raw, ["pose", "model_complexity"], 1), 1)
	pose_det_conf = _as_float(_deep_get(raw, ["pose", "min_detection_confidence"], 0.5), 0.5)
	pose_track_conf = _as_float(_deep_get(raw, ["pose", "min_tracking_confidence"], 0.5), 0.5)

	export_default_s = _as_float(_deep_get(raw, ["export", "default_seconds"], 30.0), 30.0)
	export_jump_half_s = _as_float(_deep_get(raw, ["export", "jump_fallback_half_window_seconds"], 0.75), 0.75)

	clip_poll_s = _as_float(_deep_get(raw, ["clip_worker", "poll_seconds"], 0.5), 0.5)

	collector_q = _as_int(_deep_get(raw, ["collector", "payload_queue_maxsize"], 600), 600)
	collector_warmup = _as_float(_deep_get(raw, ["collector", "calib_warmup_seconds"], 1.0), 1.0)
	collector_min_samples = _as_int(_deep_get(raw, ["collector", "calib_min_samples"], 6), 6)
	collector_alpha = _as_float(_deep_get(raw, ["collector", "offset_alpha"], 0.02), 0.02)
	collector_backoff0 = _as_float(_deep_get(raw, ["collector", "initial_backoff_seconds"], 0.5), 0.5)
	collector_backoff_max = _as_float(_deep_get(raw, ["collector", "max_backoff_seconds"], 8.0), 8.0)
	collector_backoff_mul = _as_float(_deep_get(raw, ["collector", "backoff_multiplier"], 1.6), 1.6)
	collector_connect_timeout = _as_float(_deep_get(raw, ["collector", "connect_timeout_seconds"], 25.0), 25.0)
	collector_stale_timeout = _as_float(_deep_get(raw, ["collector", "notify_stale_timeout_seconds"], 10.0), 10.0)
	collector_stat_interval = _as_float(_deep_get(raw, ["collector", "stat_emit_interval_seconds"], 1.0), 1.0)
	collector_sub_ref = _as_int(_deep_get(raw, ["collector", "subscribe_ref_id"], 99), 99)
	collector_hello_ref = _as_int(_deep_get(raw, ["collector", "hello_ref_id"], 1), 1)

	ble_scan_timeout = _as_float(_deep_get(raw, ["ble", "scan_timeout_seconds"], 7.0), 7.0)
	ble_find_timeout = _as_float(_deep_get(raw, ["ble", "connect_find_timeout_seconds"], 10.0), 10.0)
	ble_discovery_timeout = _as_float(_deep_get(raw, ["ble", "connect_discovery_timeout_seconds"], 7.0), 7.0)
	ble_fallback_scan_timeout = _as_float(
		_deep_get(raw, ["ble", "connect_fallback_scan_timeout_seconds"], 7.0),
		7.0,
	)
	ble_refind_timeout = _as_float(_deep_get(raw, ["ble", "connect_refind_timeout_seconds"], 5.0), 5.0)

	api_jumps_limit = _as_int(_deep_get(raw, ["api", "jumps_list_limit_default"], 200), 200)
	api_jumps_limit_max = _as_int(_deep_get(raw, ["api", "jumps_list_limit_max"], 1000), 1000)
	api_pages_jumps_limit = _as_int(_deep_get(raw, ["api", "pages_jumps_preload_limit"], 200), 200)
	api_frames_limit = _as_int(_deep_get(raw, ["api", "session_frames_limit_default"], 200000), 200000)

	http_status_timeout = _as_float(_deep_get(raw, ["http_proxy", "status_timeout_seconds"], 2.0), 2.0)
	http_post_timeout = _as_float(_deep_get(raw, ["http_proxy", "post_timeout_seconds"], 5.0), 5.0)
	http_mjpeg_open_timeout = _as_float(
		_deep_get(raw, ["http_proxy", "mjpeg_open_timeout_seconds"], 10.0),
		10.0,
	)
	http_snapshot_timeout = _as_float(_deep_get(raw, ["http_proxy", "snapshot_timeout_seconds"], 3.0), 3.0)
	http_mjpeg_chunk_bytes = _as_int(_deep_get(raw, ["http_proxy", "mjpeg_read_chunk_bytes"], 8192), 8192)

	rt_overlap_tol = _as_float(_deep_get(raw, ["runtime", "overlap_tolerance_seconds"], 0.1), 0.1)
	rt_min_window = _as_float(_deep_get(raw, ["runtime", "min_window_size_seconds"], 2.0), 2.0)
	rt_fallback_half = _as_float(_deep_get(raw, ["runtime", "fallback_window_half_seconds"], 1.0), 1.0)
	rt_wait_poll = _as_float(_deep_get(raw, ["runtime", "imu_wait_poll_seconds"], 0.02), 0.02)
	rt_wait_slack = _as_float(_deep_get(raw, ["runtime", "imu_wait_slack_seconds"], 0.5), 0.5)
	rt_backfill_jumps = _as_int(_deep_get(raw, ["runtime", "backfill_jumps_limit"], 2000), 2000)
	rt_backfill_frames = _as_int(_deep_get(raw, ["runtime", "backfill_frames_limit"], 500000), 500000)
	rt_jump_worker_idle = _as_float(_deep_get(raw, ["runtime", "jump_worker_idle_sleep_seconds"], 0.1), 0.1)
	rt_frame_sync_interval = _as_float(_deep_get(raw, ["runtime", "frame_sync_interval_seconds"], 0.1), 0.1)
	rt_frame_sync_backoff = _as_float(_deep_get(raw, ["runtime", "frame_sync_error_backoff_seconds"], 0.5), 0.5)
	rt_video_start_wait = _as_float(_deep_get(raw, ["runtime", "video_start_wait_seconds"], 3.0), 3.0)
	rt_video_poll = _as_float(_deep_get(raw, ["runtime", "video_status_poll_seconds"], 0.1), 0.1)
	rt_video_rec_delay = _as_float(
		_deep_get(raw, ["runtime", "video_recording_poll_delay_seconds"], 0.05),
		0.05,
	)
	rt_video_connect_warmup = _as_float(
		_deep_get(raw, ["runtime", "video_connect_warmup_seconds"], 0.2),
		0.2,
	)
	rt_session_video_wait = _as_float(
		_deep_get(raw, ["runtime", "session_video_wait_timeout_seconds"], 15.0),
		15.0,
	)
	rt_session_video_poll = _as_float(
		_deep_get(raw, ["runtime", "session_video_wait_poll_seconds"], 0.25),
		0.25,
	)
	rt_edge_guard = _as_float(_deep_get(raw, ["runtime", "takeoff_landing_edge_guard_seconds"], 0.5), 0.5)
	rt_clip_extra = _as_float(_deep_get(raw, ["runtime", "clip_fallback_extra_seconds"], 0.4), 0.4)
	rt_clip_wait_mp4 = _as_float(_deep_get(raw, ["runtime", "clip_wait_mp4_timeout_seconds"], 900.0), 900.0)
	rt_jump_windows_history = _as_int(_deep_get(raw, ["runtime", "jump_windows_history_per_session"], 100), 100)
	rt_imu_rate_window = _as_float(_deep_get(raw, ["runtime", "imu_rx_rate_window_seconds"], 5.0), 5.0)
	rt_imu_prune_min_len = _as_int(_deep_get(raw, ["runtime", "imu_history_prune_check_min_len"], 100), 100)
	rt_subprocess_wait = _as_float(_deep_get(raw, ["runtime", "subprocess_wait_timeout_seconds"], 3.0), 3.0)
	rt_picamera_join_timeout = _as_float(
		_deep_get(raw, ["runtime", "picamera_thread_join_timeout_seconds"], 2.0),
		2.0,
	)
	rt_picamera_loop_sleep = _as_float(
		_deep_get(raw, ["runtime", "picamera_run_loop_sleep_seconds"], 0.05),
		0.05,
	)

	video_backend = _as_str(_deep_get(raw, ["video", "backend"], "picamera2"), "picamera2").strip().lower()
	proc_enabled = _as_bool(_deep_get(raw, ["video", "process", "enabled"], False), False)
	proc_host = _as_str(_deep_get(raw, ["video", "process", "collector_host"], "127.0.0.1"), "127.0.0.1")
	proc_port = _as_int(_deep_get(raw, ["video", "process", "collector_port"], 18081), 18081)
	video_default_mjpeg_fps = _as_float(_deep_get(raw, ["video", "default_mjpeg_fps"], 15.0), 15.0)
	video_recording_fps = _as_int(_deep_get(raw, ["video", "recording_fps"], 30), 30)
	video_preview_jpeg_quality = _as_int(_deep_get(raw, ["video", "preview_jpeg_quality"], 80), 80)

	pc2_primary_idx = _as_int(_deep_get(raw, ["video", "picamera2", "primary_index"], 1), 1)
	pc2_gs_idx = _as_int(_deep_get(raw, ["video", "picamera2", "gs_index"], 0), 0)
	pc2_module3 = _parse_camera_cfg(_deep_get(raw, ["video", "picamera2", "module3"], {}))
	pc2_gs = _parse_camera_cfg(_deep_get(raw, ["video", "picamera2", "gs"], {}))

	return AppConfig(
		database=DatabaseConfig(
			url=db_url,
			pool_min_size=max(1, int(db_pool_min)),
			pool_max_size=max(max(1, int(db_pool_min)), int(db_pool_max)),
		),
		movesense=MovesenseConfig(
			default_device=m_default_device,
			default_mode=m_default_mode,
			default_rate=int(m_default_rate) if int(m_default_rate) > 0 else 104,
			ble_adapter=m_ble_adapter,
		),
		imu_udp=ImuUdpConfig(host=udp_host, port=int(udp_port) if int(udp_port) > 0 else 9999),
		auto_connect=AutoConnectConfig(
			imu_retry_interval_seconds=float(imu_retry_sec),
			jump_detection_enabled=bool(auto_jump_detection),
		),
		buffers=BuffersConfig(
			imu_history_seconds=max(1.0, float(imu_hist_s)),
			frame_history_seconds=max(1.0, float(frame_hist_s)),
			jump_sample_queue_maxsize=max(10, int(jump_q_max)),
			jump_events_maxlen=max(10, int(jump_events_maxlen)),
		),
		jobs=JobsConfig(jump_clip_jobs_dir=jobs_dir),
		sessions=SessionsConfig(base_dir=sessions_base_dir, jump_clips_subdir=jump_clips_subdir),
		video=VideoConfig(
			backend=video_backend or "picamera2",
			process=VideoProcessConfig(enabled=proc_enabled, collector_host=proc_host, collector_port=int(proc_port)),
			default_mjpeg_fps=max(1.0, float(video_default_mjpeg_fps)),
			recording_fps=max(1, int(video_recording_fps)),
			preview_jpeg_quality=max(1, min(100, int(video_preview_jpeg_quality))),
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
		jump_detection=JumpDetectionConfig(
			window_seconds=max(1.0, float(jd_window)),
			min_peak_height_m_s2=max(0.0, float(jd_min_peak)),
			min_peak_distance_s=max(0.0, float(jd_peak_dist)),
			min_flight_time_s=max(0.0, float(jd_min_flight)),
			max_flight_time_s=max(float(jd_min_flight), float(jd_max_flight)),
			min_jump_height_m=max(0.0, float(jd_min_jump_h)),
			min_jump_peak_az_no_g=max(0.0, float(jd_min_az)),
			min_jump_peak_gz_deg_s=max(0.0, float(jd_min_gz)),
			min_new_event_separation_s=max(0.0, float(jd_min_sep)),
			min_takeoff_to_peak_s=max(0.0, float(jd_min_to_peak)),
			max_takeoff_to_peak_s=max(max(0.0, float(jd_min_to_peak)), float(jd_max_to_peak)),
			min_peak_to_landing_s=max(0.0, float(jd_min_peak_to_land)),
			max_peak_to_landing_s=max(max(0.0, float(jd_min_peak_to_land)), float(jd_max_peak_to_land)),
			min_revs=max(0.0, float(jd_min_revs)),
			analysis_interval_s=max(0.05, float(jd_interval)),
			smoothing_cutoff_hz=max(0.1, float(jd_smoothing_cutoff)),
			refine_window_s=max(0.01, float(jd_refine_window)),
			bias_window_start_s=max(0.0, float(jd_bias_start)),
			bias_window_end_s=max(0.0, float(jd_bias_end)),
		),
		pose=PoseConfig(
			max_fps=max(1.0, float(pose_max_fps)),
			model_complexity=max(0, min(2, int(pose_model))),
			min_detection_confidence=max(0.0, min(1.0, float(pose_det_conf))),
			min_tracking_confidence=max(0.0, min(1.0, float(pose_track_conf))),
		),
		export=ExportConfig(
			default_seconds=max(0.1, float(export_default_s)),
			jump_fallback_half_window_seconds=max(0.0, float(export_jump_half_s)),
		),
		clip_worker=ClipWorkerConfig(
			poll_seconds=max(0.01, float(clip_poll_s)),
		),
		collector=CollectorConfig(
			payload_queue_maxsize=max(10, int(collector_q)),
			calib_warmup_seconds=max(0.0, float(collector_warmup)),
			calib_min_samples=max(1, int(collector_min_samples)),
			offset_alpha=max(0.0, min(1.0, float(collector_alpha))),
			initial_backoff_seconds=max(0.01, float(collector_backoff0)),
			max_backoff_seconds=max(0.01, float(collector_backoff_max)),
			backoff_multiplier=max(1.0, float(collector_backoff_mul)),
			connect_timeout_seconds=max(0.1, float(collector_connect_timeout)),
			notify_stale_timeout_seconds=max(0.1, float(collector_stale_timeout)),
			stat_emit_interval_seconds=max(0.1, float(collector_stat_interval)),
			subscribe_ref_id=max(1, int(collector_sub_ref)),
			hello_ref_id=max(1, int(collector_hello_ref)),
		),
		ble=BleConfig(
			scan_timeout_seconds=max(0.1, float(ble_scan_timeout)),
			connect_find_timeout_seconds=max(0.1, float(ble_find_timeout)),
			connect_discovery_timeout_seconds=max(0.1, float(ble_discovery_timeout)),
			connect_fallback_scan_timeout_seconds=max(0.1, float(ble_fallback_scan_timeout)),
			connect_refind_timeout_seconds=max(0.1, float(ble_refind_timeout)),
		),
		api=ApiConfig(
			jumps_list_limit_default=max(1, int(api_jumps_limit)),
			jumps_list_limit_max=max(max(1, int(api_jumps_limit)), int(api_jumps_limit_max)),
			pages_jumps_preload_limit=max(1, int(api_pages_jumps_limit)),
			session_frames_limit_default=max(1, int(api_frames_limit)),
		),
		http_proxy=HttpProxyConfig(
			status_timeout_seconds=max(0.1, float(http_status_timeout)),
			post_timeout_seconds=max(0.1, float(http_post_timeout)),
			mjpeg_open_timeout_seconds=max(0.1, float(http_mjpeg_open_timeout)),
			snapshot_timeout_seconds=max(0.1, float(http_snapshot_timeout)),
			mjpeg_read_chunk_bytes=max(256, int(http_mjpeg_chunk_bytes)),
		),
		runtime=RuntimeConfig(
			overlap_tolerance_seconds=max(0.0, float(rt_overlap_tol)),
			min_window_size_seconds=max(0.1, float(rt_min_window)),
			fallback_window_half_seconds=max(0.1, float(rt_fallback_half)),
			imu_wait_poll_seconds=max(0.001, float(rt_wait_poll)),
			imu_wait_slack_seconds=max(0.0, float(rt_wait_slack)),
			backfill_jumps_limit=max(1, int(rt_backfill_jumps)),
			backfill_frames_limit=max(1, int(rt_backfill_frames)),
			jump_worker_idle_sleep_seconds=max(0.001, float(rt_jump_worker_idle)),
			frame_sync_interval_seconds=max(0.001, float(rt_frame_sync_interval)),
			frame_sync_error_backoff_seconds=max(0.001, float(rt_frame_sync_backoff)),
			video_start_wait_seconds=max(0.1, float(rt_video_start_wait)),
			video_status_poll_seconds=max(0.001, float(rt_video_poll)),
			video_recording_poll_delay_seconds=max(0.001, float(rt_video_rec_delay)),
			video_connect_warmup_seconds=max(0.001, float(rt_video_connect_warmup)),
			session_video_wait_timeout_seconds=max(0.1, float(rt_session_video_wait)),
			session_video_wait_poll_seconds=max(0.001, float(rt_session_video_poll)),
			takeoff_landing_edge_guard_seconds=max(0.0, float(rt_edge_guard)),
			clip_fallback_extra_seconds=max(0.0, float(rt_clip_extra)),
			clip_wait_mp4_timeout_seconds=max(0.1, float(rt_clip_wait_mp4)),
			jump_windows_history_per_session=max(1, int(rt_jump_windows_history)),
			imu_rx_rate_window_seconds=max(0.1, float(rt_imu_rate_window)),
			imu_history_prune_check_min_len=max(1, int(rt_imu_prune_min_len)),
			subprocess_wait_timeout_seconds=max(0.1, float(rt_subprocess_wait)),
			picamera_thread_join_timeout_seconds=max(0.1, float(rt_picamera_join_timeout)),
			picamera_run_loop_sleep_seconds=max(0.001, float(rt_picamera_loop_sleep)),
		),
	)


def get_config() -> AppConfig:
	global _CONFIG_CACHE
	if _CONFIG_CACHE is None:
		_CONFIG_CACHE = load_config()
	return _CONFIG_CACHE


def get_jump_detection_defaults(cfg: Optional[AppConfig] = None) -> Dict[str, float]:
	"""Return jump-detection defaults as a mutable dict for runtime/DB/UI use."""
	c = cfg or get_config()
	jd = c.jump_detection
	return {
		"window_seconds": float(jd.window_seconds),
		"min_jump_height_m": float(jd.min_jump_height_m),
		"min_peak_height_m_s2": float(jd.min_peak_height_m_s2),
		"min_peak_distance_s": float(jd.min_peak_distance_s),
		"min_flight_time_s": float(jd.min_flight_time_s),
		"max_flight_time_s": float(jd.max_flight_time_s),
		"min_jump_peak_az_no_g": float(jd.min_jump_peak_az_no_g),
		"min_jump_peak_gz_deg_s": float(jd.min_jump_peak_gz_deg_s),
		"min_new_event_separation_s": float(jd.min_new_event_separation_s),
		"min_takeoff_to_peak_s": float(jd.min_takeoff_to_peak_s),
		"max_takeoff_to_peak_s": float(jd.max_takeoff_to_peak_s),
		"min_peak_to_landing_s": float(jd.min_peak_to_landing_s),
		"max_peak_to_landing_s": float(jd.max_peak_to_landing_s),
		"analysis_interval_s": float(jd.analysis_interval_s),
		"min_revs": float(jd.min_revs),
		"smoothing_cutoff_hz": float(jd.smoothing_cutoff_hz),
		"refine_window_s": float(jd.refine_window_s),
		"bias_window_start_s": float(jd.bias_window_start_s),
		"bias_window_end_s": float(jd.bias_window_end_s),
	}


