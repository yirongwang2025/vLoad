"""
Shared state for routers. Populated by server.py at startup.
Routers import from here to avoid circular imports.
"""
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Set

# Lazy HTML loader (set by server): get_page_html(filename) -> str, loads on first request (B.1)
get_page_html: Optional[Callable[[str], str]] = None
UI_DIR: Optional[Path] = None

# WebSocket manager (set by server after creating ConnectionManager)
manager: Any = None

# Session/video/frame state (set by server in lifespan or at module level)
_session_id: Optional[str] = None
_session_dir: Any = None
_session_lock: Any = None
_imu_csv_fh: Any = None
_video: Any = None
_frame_history: Optional[Deque[Dict[str, Any]]] = None
_dbg: Dict[str, Any] = {}
CFG: Any = None
_active_mode: str = ""
_active_rate: int = 104
_jump_config: Dict[str, float] = {}
_jump_sample_queue: Any = None
_jump_detection_enabled: bool = False

# Helpers (set by server)
_log_to_clients: Any = None
_session_base_dir: Any = None
_enqueue_jump_clip_job: Any = None
_maybe_schedule_pose_for_jump: Any = None
_run_pose_for_jump_best_effort: Any = None
_count_clip_jobs_pending: Any = None
_count_clip_jobs_done: Any = None
_count_clip_jobs_failed: Any = None
_read_last_clip_job_error: Any = None

# Pose job tracking (can live here so pose_runner and server both use it)
_pose_jobs_inflight: Set[int] = set()
