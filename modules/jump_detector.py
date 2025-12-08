import time
from collections import deque
from typing import Deque, Dict, Any, Callable, List, Optional

from .web_jump_detection import (
	preprocess_vertical_series,
	find_vertical_peaks,
	build_jump_windows,
)


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

		# Phase‑tracking flags
		self._phase1_logged: bool = False
		self._phase2_1_logged: bool = False
		self._last_phase2_2_log_t: float = 0.0
		self._last_phase2_3_log_t: float = 0.0

		# Phase 2.2 parameters: simple, conservative defaults.
		# Minimum |az-g| needed to count as a peak (m/s^2).
		self.min_peak_height_m_s2: float = 3.0
		# Minimum spacing between peaks (seconds).
		self.min_peak_distance_s: float = 0.18

		# Phase 2.3 parameters: candidate jump windows.
		# Plausible flight-time window (seconds).
		self.min_flight_time_s: float = 0.25
		self.max_flight_time_s: float = 0.80

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

			# Phase 2.2: periodically (about once per second) count the number
			# of vertical peaks in the current buffer using a simple, SciPy‑
			# free peak finder. This is still diagnostic only; no jumps yet.
			now_wall = time.time()
			if self._phase2_1_logged and now_wall - self._last_phase2_2_log_t >= 1.0:
				stats = preprocess_vertical_series(list(self._buffer))
				min_dist_samples = max(
					1, int(self.min_peak_distance_s * self.sample_rate_hz)
				)
				peak_indices = find_vertical_peaks(
					stats,
					min_height=self.min_peak_height_m_s2,
					min_distance_samples=min_dist_samples,
				)
				msg = (
					f"[Phase2.2] Peaks in ~{stats['duration']:.2f} s window: "
					f"{len(peak_indices)} (min_height={self.min_peak_height_m_s2:.1f} m/s², "
					f"min_spacing={self.min_peak_distance_s:.2f} s)"
				)
				self.logger(msg)
				self._last_phase2_2_log_t = now_wall

			# Phase 2.3: at a slower cadence (~every 2 seconds), build simple
			# candidate jump windows from the peak indices and log their
			# flight times. This still does not emit "real" jumps; it is
			# purely diagnostic.
			if self._phase2_1_logged and now_wall - self._last_phase2_3_log_t >= 2.0:
				stats = preprocess_vertical_series(list(self._buffer))
				min_dist_samples = max(
					1, int(self.min_peak_distance_s * self.sample_rate_hz)
				)
				peak_indices = find_vertical_peaks(
					stats,
					min_height=self.min_peak_height_m_s2,
					min_distance_samples=min_dist_samples,
				)
				windows = build_jump_windows(
					stats,
					peak_indices,
					min_flight_time_s=self.min_flight_time_s,
					max_flight_time_s=self.max_flight_time_s,
				)
				if windows:
					flight_times = ", ".join(
						f"{w['flight_time']:.3f}" for w in windows[:5]
					)
				else:
					flight_times = ""
				msg = (
					f"[Phase2.3] Candidate windows in ~{stats['duration']:.2f} s: "
					f"{len(windows)}"
				)
				if flight_times:
					msg += f" (T_f: {flight_times} s{' ...' if len(windows) > 5 else ''})"
				self.logger(msg)
				self._last_phase2_3_log_t = now_wall
		except Exception:
			# Never let Phase 1 diagnostics break the main streaming path.
			return []

		# Phase 1: no jump events yet.
		return []


