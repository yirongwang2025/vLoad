import time
from collections import deque
from typing import Deque, Dict, Any, Callable, List, Optional

from .web_jump_detection import (
	preprocess_vertical_series,
	find_vertical_peaks,
	build_jump_windows,
	compute_window_metrics,
	select_jump_events,
)


class JumpDetectorRealtime:
	"""
	Realtime vertical acceleration / gyro tracker + Bruening‑style jump detector.

	Phases:
	  - Phase 1:
	      Keep a short rolling window of samples (acc_z, gyro_z) and emit a
	      one‑time log message so we can verify vertical tracking end‑to‑end.
	  - Phase 2.1–2.4:
	      Periodically log diagnostics on preprocessing, vertical peaks,
	      candidate jump windows, and basic jump metrics (height, ωz).
	  - Phase 2.5/2.6:
	      Promote enriched windows to higher‑confidence jump events using
	      select_jump_events(...), log concise [Jump] lines, and return the
	      new events so the server can forward them to the UI.

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
		config: Optional[Dict[str, float]] = None,
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
		self._last_phase2_4_log_t: float = 0.0

		# Phase 2.2 parameters: simple, conservative defaults.
		# Minimum |az-g| needed to count as a peak (m/s^2).
		self.min_peak_height_m_s2: float = 3.0
		# Minimum spacing between peaks (seconds).
		self.min_peak_distance_s: float = 0.18

		# Phase 2.3 parameters: candidate jump windows.
		# Plausible flight-time window (seconds).
		self.min_flight_time_s: float = 0.25
		self.max_flight_time_s: float = 0.80

		# Phase 2.5 parameters: promote candidates to actual jump events.
		self.min_jump_height_m: float = 0.15
		self.min_jump_peak_az_no_g: float = 3.5
		self.min_jump_peak_gz_deg_s: float = 180.0
		self.min_new_event_separation_s: float = 0.5
		self._last_emitted_peak_t: float = 0.0

		# Optional overrides from config dict.
		if config:
			for key, value in config.items():
				if hasattr(self, key):
					try:
						setattr(self, key, float(value))
					except (TypeError, ValueError):
						continue

	def update(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""
		Ingest one IMU sample and update internal state.

		Returns a (possibly empty) list of detected jump events. In quiet
		motion this is usually [], while clear multi‑rev jumps will produce
		a small number of well‑separated events with fields such as:
		  - t_peak, flight_time, height, peak_az_no_g, peak_gz,
		    rotation_phase, confidence.
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

			# Phase 2.4 & 2.5: every few seconds, derive simple jump metrics
			# (height and peak rotation speed) for the current candidate
			# windows, log a compact summary, and then filter/score into
			# higher-confidence jump events.
			new_events: List[Dict[str, Any]] = []
			if self._phase2_1_logged and now_wall - self._last_phase2_4_log_t >= 3.0:
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
				metrics = compute_window_metrics(stats, windows)
				if metrics:
					parts = []
					for m in metrics[:3]:
						parts.append(
							f"T_f={m['flight_time']:.3f}s,"
							f"h={m['height']:.3f}m,"
							f"ωz={m['peak_gz']:.0f}°/s"
						)
					joined = " | ".join(parts)
					if len(metrics) > 3:
						joined += " ..."
					msg = f"[Phase2.4] Jump metrics: {joined}"
				else:
					msg = "[Phase2.4] Jump metrics: none in current window"
				self.logger(msg)
				self._last_phase2_4_log_t = now_wall

				# Phase 2.5 – select higher-confidence jump events.
				candidates = select_jump_events(
					metrics,
					min_height_m=self.min_jump_height_m,
					min_peak_az_no_g=self.min_jump_peak_az_no_g,
					min_peak_gz_deg_s=self.min_jump_peak_gz_deg_s,
					min_separation_s=self.min_new_event_separation_s,
				)
				for ev in candidates:
					t_peak = float(ev.get("t_peak", 0.0))
					if (
						t_peak - self._last_emitted_peak_t
						>= self.min_new_event_separation_s
					):
						new_events.append(ev)
						self._last_emitted_peak_t = t_peak

				for ev in new_events:
					self.logger(
						"[Jump] t_peak={:.3f}, T_f={:.3f}s, h={:.3f}m, "
						"ωz_peak={:.0f}°/s, phase={:.2f}, conf={:.2f}".format(
							ev.get("t_peak", 0.0),
							ev.get("flight_time", 0.0),
							ev.get("height", 0.0),
							ev.get("peak_gz", 0.0),
							ev.get("rotation_phase", 0.0),
							ev.get("confidence", 0.0),
						)
					)

		except Exception:
			# Never let Phase 1 diagnostics break the main streaming path.
			return []

		# Phase 2.5: return any new jump events detected in this call.
		return new_events


