import time
from collections import deque
from typing import Deque, Dict, Any, Callable, List, Optional

from .web_jump_detection import preprocess_vertical_series


class JumpDetectorRealtime:
	"""
	Phase 1: simple real‑time tracker for vertical acceleration / gyro.

	This does NOT yet try to detect jumps. It just keeps a short rolling
	window of samples (acc_z, gyro_z) and emits a one‑time log message so
	we can verify that vertical tracking is working end‑to‑end.

	Expected sample format passed to `update`:
	    {
	        "t": <wall-clock seconds, e.g. time.time()>,
	        "acc": [ax, ay, az],
	        "gyro": [gx, gy, gz],
	        # "mag": [...]  # optional, ignored for Phase 1
	    }
	"""

	def __init__(
		self,
		sample_rate_hz: float = 104.0,
		window_seconds: float = 3.0,
		logger: Optional[Callable[[str], None]] = None,
	) -> None:
		self.sample_rate_hz = float(sample_rate_hz) if sample_rate_hz > 0 else 104.0
		self.window_seconds = float(window_seconds) if window_seconds > 0 else 3.0
		self.logger: Callable[[str], None] = logger or (lambda _msg: None)

		maxlen = max(10, int(self.sample_rate_hz * self.window_seconds))
		self._buffer: Deque[Dict[str, float]] = deque(maxlen=maxlen)

		self._phase1_logged: bool = False
		self._phase2_1_logged: bool = False

	def update(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""
		Ingest one IMU sample and update internal state.

		Returns a (possibly empty) list of detected jump events. For Phase 1
		this is always an empty list – jump detection comes in later phases.
		"""
		try:
			acc = sample.get("acc") or []
			gyro = sample.get("gyro") or []
			if len(acc) < 3 or len(gyro) < 3:
				return []

			az = float(acc[2])
			gz = float(gyro[2])
			t = float(sample.get("t", time.time()))

			self._buffer.append({"t": t, "az": az, "gz": gz})

			# Emit a one‑time log line once we have at least ~1 second of data
			if not self._phase1_logged and len(self._buffer) >= int(self.sample_rate_hz):
				stats = preprocess_vertical_series(list(self._buffer))
				msg = (
					f"[Phase1] Vertical tracking active over ~{stats['duration']:.2f} s: "
					f"max|az|={stats['max_abs_az']:.2f} m/s², "
					f"max|az-g|={stats['max_abs_az_no_g']:.2f} m/s², "
					f"max|gz|={stats['max_abs_gz']:.1f} °/s"
				)
				self.logger(msg)
				self._phase1_logged = True

			# Phase 2.1: once the window is reasonably full, log a brief
			# preprocessing confirmation using the same helper. This is kept
			# to a single message per connection to avoid log spam.
			if (
				not self._phase2_1_logged
				and len(self._buffer) >= int(self.sample_rate_hz * self.window_seconds * 0.9)
			):
				stats = preprocess_vertical_series(list(self._buffer))
				msg = (
					f"[Phase2.1] Preprocess buffer: {stats['count']} samples over "
					f"~{stats['duration']:.2f} s, "
					f"max|az-g|={stats['max_abs_az_no_g']:.2f} m/s², "
					f"max|gz|={stats['max_abs_gz']:.1f} °/s"
				)
				self.logger(msg)
				self._phase2_1_logged = True
		except Exception:
			# Never let Phase 1 diagnostics break the main streaming path.
			return []

		# Phase 1: no jump events yet.
		return []


