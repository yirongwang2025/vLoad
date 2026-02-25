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
from .config import get_config, get_jump_detection_defaults


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
		sample_rate_hz: Optional[float] = None,
		window_seconds: Optional[float] = None,
		logger: Optional[Callable[[str], None]] = None,
		config: Optional[Dict[str, float]] = None,
	) -> None:
		cfg = get_config()
		rate_default = float(cfg.movesense.default_rate)
		window_default = float(cfg.jump_detection.window_seconds)
		self.sample_rate_hz = (
			float(sample_rate_hz)
			if (sample_rate_hz is not None and float(sample_rate_hz) > 0.0)
			else rate_default
		)
		self.window_seconds = (
			float(window_seconds)
			if (window_seconds is not None and float(window_seconds) > 0.0)
			else window_default
		)
		self.logger: Callable[[str], None] = logger or (lambda _msg: None)

		maxlen = max(10, int(self.sample_rate_hz * self.window_seconds))
		self._buffer: Deque[Dict[str, float]] = deque(maxlen=maxlen)

		# Phase‑tracking flags
		self._phase1_logged: bool = False
		self._phase2_1_logged: bool = False
		self._last_phase2_2_log_t: float = 0.0
		self._last_phase2_3_log_t: float = 0.0
		self._last_phase2_4_log_t: float = 0.0
		self._last_debug_diag_t: float = 0.0

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
		self.min_takeoff_to_peak_s: float = 0.02
		self.max_takeoff_to_peak_s: float = 0.80
		self.min_peak_to_landing_s: float = 0.02
		self.max_peak_to_landing_s: float = 0.80
		# Step 2.3: only treat events with at least this many revolutions as real jumps.
		# Uses revolutions_est computed in compute_window_metrics (Step 2.2).
		self.min_revs: float = 0.0
		# How often (in sample-time seconds) to run the heavier window+metrics+event selection.
		# Lower values reduce detection latency but increase CPU. 0.5s is a good default for live use.
		self.analysis_interval_s: float = 0.5
		self._last_emitted_peak_t: float = 0.0
		# Persistent tracking of last takeoff time across analysis cycles for separation enforcement
		self._last_t_takeoff_persistent: float = -1e9

		# Pull centralized defaults from config.json first.
		default_cfg = get_jump_detection_defaults(get_config())
		for key, value in default_cfg.items():
			if hasattr(self, key):
				try:
					setattr(self, key, float(value))
				except (TypeError, ValueError):
					continue

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
			gx = float(gyro[0])
			gy = float(gyro[1])
			gz = float(gyro[2])
			t = float(sample.get("t", time.time()))

			self._buffer.append({"t": t, "az": az, "gx": gx, "gy": gy, "gz": gz})

			# Emit a one‑time log line once we have at least ~1 second of data
			if not self._phase1_logged and len(self._buffer) >= int(self.sample_rate_hz):
				# Use list() only once - buffer is deque, need list for preprocessing
				buffer_list = list(self._buffer)
				stats = preprocess_vertical_series(buffer_list)
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
			# if (
			# 	not self._phase2_1_logged
			# 	and len(self._buffer) >= int(self.sample_rate_hz * self.window_seconds * 0.9)
			# ):
				# Phase 2.1 diagnostic logging removed per user request.
				# self._phase2_1_logged = True

			# Phase 2.2: periodically (about once per second) count the number
			# of vertical peaks in the current buffer using a simple, SciPy‑
			# free peak finder. This is still diagnostic only; no jumps yet.
			# Use sample-time for cadence decisions so backlog / timestamp alignment doesn't
			# accidentally change the analysis frequency.
			now_wall = t
			# if self._phase2_1_logged and now_wall - self._last_phase2_2_log_t >= 1.0:
				# Phase 2.2 diagnostic logging removed per user request.
				# self._last_phase2_2_log_t = now_wall

			# Phase 2.3: at a slower cadence (~every 2 seconds), update tracking timestamp.
			# (Diagnostic logging removed per user request.)
			#if self._phase2_1_logged and now_wall - self._last_phase2_3_log_t >= 2.0:
				# self._last_phase2_3_log_t = now_wall

			# Phase 2.4 & 2.5: every few seconds, derive simple jump metrics
			# (height and peak rotation speed) for the current candidate
			# windows, and then filter/score into higher-confidence jump events.
			# (Phase 2.4 diagnostic logging removed per user request.)
			new_events: List[Dict[str, Any]] = []
			if now_wall - self._last_phase2_4_log_t >= float(self.analysis_interval_s):
				# Convert deque to list only once per analysis cycle
				buffer_list = list(self._buffer)
				stats = preprocess_vertical_series(buffer_list)
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
				# Phase 2.4 diagnostic logging removed per user request.
				self._last_phase2_4_log_t = now_wall

				# Phase 2.5 – select higher-confidence jump events.
				# Pass persistent last_t_takeoff to enforce separation across analysis cycles
				candidates, updated_last_t_takeoff = select_jump_events(
					metrics,
					min_height_m=self.min_jump_height_m,
					min_peak_az_no_g=self.min_jump_peak_az_no_g,
					min_peak_gz_deg_s=self.min_jump_peak_gz_deg_s,
					min_separation_s=self.min_new_event_separation_s,
					last_t_takeoff_persistent=self._last_t_takeoff_persistent,
				)
				# Update persistent tracking for next analysis cycle
				self._last_t_takeoff_persistent = updated_last_t_takeoff
				if now_wall - self._last_debug_diag_t >= 1.0:
					try:
						# Keep the cadence checkpoint without emitting debug logs.
						_ = (
							max((float(m.get("peak_az_no_g", 0.0)) for m in metrics), default=0.0),
							max((float(m.get("peak_gz", 0.0)) for m in metrics), default=0.0),
							max((float(m.get("height", 0.0)) for m in metrics), default=0.0),
						)
					except Exception:
						pass
					self._last_debug_diag_t = now_wall
				# Temporal-spacing filter for takeoff/peak/landing geometry.
				spacing_filtered: List[Dict[str, Any]] = []
				for ev in candidates:
					try:
						t_takeoff = float(ev.get("t_takeoff", 0.0))
						t_peak = float(ev.get("t_peak", t_takeoff))
						t_landing = float(ev.get("t_landing", t_peak))
					except (TypeError, ValueError):
						continue
					d_tp = t_peak - t_takeoff
					d_pl = t_landing - t_peak
					if (
						d_tp < float(self.min_takeoff_to_peak_s)
						or d_tp > float(self.max_takeoff_to_peak_s)
						or d_pl < float(self.min_peak_to_landing_s)
						or d_pl > float(self.max_peak_to_landing_s)
					):
						continue
					spacing_filtered.append(ev)
				candidates = spacing_filtered
				# Step 2.3: filter by minimum revolutions (if available).
				if self.min_revs and self.min_revs > 0.0:
					filtered: List[Dict[str, Any]] = []
					for ev in candidates:
						try:
							rev_est = float(ev.get("revolutions_est", 0.0))
						except (TypeError, ValueError):
							rev_est = 0.0
						if rev_est >= float(self.min_revs):
							filtered.append(ev)
					candidates = filtered
				# Additional debouncing: filter by last emitted PEAK time.
				# Using t_takeoff here can let near-duplicate events pass when takeoff
				# estimates drift but the physical peak is the same.
				for ev in candidates:
					t_peak = float(ev.get("t_peak", ev.get("t_takeoff", 0.0)))
					if (
						t_peak - self._last_emitted_peak_t
						>= self.min_new_event_separation_s
					):
						new_events.append(ev)
						self._last_emitted_peak_t = t_peak
				for ev in new_events:
					# Optional Step 2.2 fields
					rev_est = ev.get("revolutions_est", None)
					underrot = ev.get("underrotation", None)
					rev_str = ""
					if isinstance(rev_est, (int, float)) and isinstance(underrot, (int, float)):
						rev_str = ", rev≈{:.2f}, UR={:.2f}".format(float(rev_est), float(underrot))
					self.logger(
						("[Jump] t_peak={:.3f}, T_f={:.3f}s, h={:.3f}m, "
						 "ωz_peak={:.0f}°/s, phase={:.2f}, conf={:.2f}{}").format(
							ev.get("t_peak", 0.0),
							ev.get("flight_time", 0.0),
							ev.get("height", 0.0),
							ev.get("peak_gz", 0.0),
							ev.get("rotation_phase", 0.0),
							ev.get("confidence", 0.0),
							rev_str,
						)
					)

		except Exception:
			# Never let Phase 1 diagnostics break the main streaming path.
			return []

		# Phase 2.5: return any new jump events detected in this call.
		return new_events
