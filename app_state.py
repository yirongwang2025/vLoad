"""
Explicit app state for 3.1 – single source of truth for runtime lifecycle.
Created in lifespan, attached to app.state.state; injected into routes via Depends(get_state).
"""
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

# Use typing types for forward refs / generics
from collections import deque


class AppState:
	"""
	Holds all runtime state for the app. Replaces globals and state module mutables.
	Populated in server lifespan; workers receive this instance as argument.
	"""
	# WebSocket and UI (set at app load)
	manager: Any = None
	get_page_html: Optional[Callable[[str], str]] = None
	UI_DIR: Optional[Path] = None

	# Session / recording
	session_id: Optional[str] = None
	session_dir: Optional[Path] = None
	session_lock: Any = None
	imu_csv_fh: Any = None
	video: Any = None
	detection_session_id: Optional[str] = None

	# Frame/IMU buffers (set in lifespan)
	frame_history: Optional[Deque[Dict[str, Any]]] = None
	imu_history: Optional[Deque[Dict[str, Any]]] = None

	# Config and active connection
	cfg: Any = None
	active_mode: str = ""
	active_rate: int = 104
	jump_config: Dict[str, float] = {}
	jump_detection_enabled: bool = False

	# Jump pipeline (set in lifespan)
	jump_sample_queue: Any = None
	jump_events: Optional[Deque[Dict[str, Any]]] = None
	next_event_id: int = 1
	jump_annotations: Dict[int, Dict[str, Any]]
	jump_windows_by_session: Dict[str, List[Tuple[int, float, float]]]

	# Debug counters
	dbg: Dict[str, Any]

	# Helpers (callables set in server after creation)
	session_base_dir: Optional[Callable[[str], Path]] = None
	log_to_clients: Any = None
	enqueue_jump_clip_job: Any = None
	run_pose_for_jump_best_effort: Any = None
	maybe_schedule_pose_for_jump: Any = None
	count_clip_jobs_pending: Any = None
	count_clip_jobs_done: Any = None
	count_clip_jobs_failed: Any = None
	read_last_clip_job_error: Any = None

	# Pose job tracking
	pose_jobs_inflight: Set[int]

	# Process/task refs (set in lifespan; used for cleanup and connect/disconnect)
	imu_proc: Any = None
	imu_udp_transport: Any = None
	jump_worker_task: Any = None
	frame_sync_task: Any = None
	clip_worker_proc: Any = None
	video_proc: Any = None

	# Rolling receive-rate window (server-side)
	imu_rx_window: Optional[Deque[Tuple[float, int]]] = None
	imu_rx_window_pkts: Optional[Deque[Tuple[float, int]]] = None

	def __init__(self) -> None:
		self.jump_annotations = {}
		self.jump_windows_by_session = {}
		self.dbg = {}
		self.pose_jobs_inflight = set()
