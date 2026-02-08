"""
Phase 2.x – utilities for vertical acceleration / gyro processing.

This module currently provides:
  - Phase 2.1: preprocessing of vertical series (including simple smoothing).
  - Phase 2.2: peak finding on the gravity‑removed vertical signal.
  - Phase 2.3: utilities for pairing peaks into candidate jump windows and
    computing simple flight‑time information.
  - Phase 2.4: utilities for deriving basic height / rotation metrics for
    each candidate window.
  - Phase 2.5/2.6: utilities for filtering and scoring jump candidates into
    higher‑confidence jump events.
"""

import math
from typing import Dict, Any, List, Optional, Tuple

# Cache for preprocess_vertical_series to avoid recomputing when buffer unchanged
_preprocess_cache: Optional[Tuple[int, float, float, Dict[str, Any]]] = None


def _interp_time_at_zero(t0: float, v0: float, t1: float, v1: float) -> float:
	"""
	Linear interpolation to estimate the time at which a signal crosses 0 between
	(t0, v0) and (t1, v1). If the segment is degenerate, returns t0.
	"""
	dv = v1 - v0
	if dv == 0.0:
		return t0
	alpha = (0.0 - v0) / dv
	# Clamp in case of tiny numeric issues
	if alpha < 0.0:
		alpha = 0.0
	elif alpha > 1.0:
		alpha = 1.0
	return t0 + alpha * (t1 - t0)


def _find_crossing_time(
	t: List[float],
	v: List[float],
	start_idx: int,
	end_idx: int,
	crossing: str,
) -> Optional[float]:
	"""
	Find an approximate (interpolated) time when v crosses 0 between indices
	[start_idx, end_idx] inclusive.

	crossing:
	  - "pos_to_neg": find first transition from >=0 to <0
	  - "neg_to_pos": find first transition from <0 to >=0

	Returns None if not found or if indices are invalid.
	"""
	n = min(len(t), len(v))
	if n < 2:
		return None
	i0 = max(0, min(int(start_idx), n - 1))
	i1 = max(0, min(int(end_idx), n - 1))
	if i0 == i1:
		return None

	step = 1 if i1 > i0 else -1
	i = i0
	while i != i1:
		j = i + step
		if j < 0 or j >= n:
			break
		v_i = float(v[i])
		v_j = float(v[j])

		if crossing == "pos_to_neg":
			# >=0 -> <0
			if v_i >= 0.0 and v_j < 0.0:
				return _interp_time_at_zero(float(t[i]), v_i, float(t[j]), v_j)
		elif crossing == "neg_to_pos":
			# <0 -> >=0
			if v_i < 0.0 and v_j >= 0.0:
				return _interp_time_at_zero(float(t[i]), v_i, float(t[j]), v_j)
		i = j

	return None


def _median(values: List[float]) -> float:
	"""Compute median of a list of floats without numpy; returns 0.0 for empty."""
	if not values:
		return 0.0
	s = sorted(float(x) for x in values)
	mid = len(s) // 2
	if len(s) % 2 == 1:
		return float(s[mid])
	return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _integrate_trapz(t: List[float], y: List[float], i0: int, i1: int) -> float:
	"""
	Trapezoidal integration of y(t) from indices [i0, i1] inclusive.
	Assumes t is in seconds and y is in units per second.
	Returns integral in y-units.
	"""
	n = min(len(t), len(y))
	if n < 2:
		return 0.0
	a = max(0, min(int(i0), n - 1))
	b = max(0, min(int(i1), n - 1))
	if b <= a:
		return 0.0
	area = 0.0
	for i in range(a, b):
		t0 = float(t[i])
		t1 = float(t[i + 1])
		dt = t1 - t0
		if dt <= 0.0:
			continue
		area += 0.5 * (float(y[i]) + float(y[i + 1])) * dt
	return float(area)


def _smooth_1pole(series: List[float], dt: float, cutoff_hz: float) -> List[float]:
	"""
	Simple first‑order low‑pass (one‑pole) filter.

	This is deliberately light‑weight and SciPy‑free; it is enough to
	reduce high‑frequency noise so that take‑off / landing structure
	is clearer, without adding much phase lag at jump‑time scales.
	"""
	if not series or dt <= 0 or cutoff_hz <= 0:
		return list(series)

	# Standard RC one‑pole: alpha = dt / (RC + dt), RC = 1 / (2π f_c)
	rc = 1.0 / (2.0 * math.pi * cutoff_hz)
	alpha = dt / (rc + dt)

	out: List[float] = [series[0]]
	for i in range(1, len(series)):
		prev = out[-1]
		curr = series[i]
		out.append(prev + alpha * (curr - prev))
	return out


def preprocess_vertical_series(buffer: List[Dict[str, float]]) -> Dict[str, Any]:
	"""
	Convert a list of samples of the form
	    {"t": <float>, "az": <float>, "gz": <float>}
	into structured series and simple summary stats.

	Returns a dict with:
	    {
	        "t": [...],
	        "az": [...],
	        "az_no_g": [...],
	        "gz": [...],
	        "count": int,
	        "duration": float,
	        "max_abs_az": float,
	        "max_abs_az_no_g": float,
	        "max_abs_gz": float,
	    }

	
	Optimization: Results are cached and only recomputed when buffer changes
	(length or first/last timestamps differ).
	"""
	global _preprocess_cache
	
	# Check cache: use cached result if buffer length and timestamps match
	if _preprocess_cache is not None:
		cached_len, cached_t0, cached_t1, cached_result = _preprocess_cache
		if len(buffer) == cached_len and buffer:
			curr_t0 = float(buffer[0].get("t", 0.0))
			curr_t1 = float(buffer[-1].get("t", 0.0))
			if curr_t0 == cached_t0 and curr_t1 == cached_t1:
				return cached_result
	
	if not buffer:
		result = {
			"t": [],
			"az": [],
			"az_no_g": [],
			"gz": [],
			"az_smooth": [],
			"az_no_g_smooth": [],
			"gz_smooth": [],
			"count": 0,
			"duration": 0.0,
			"max_abs_az": 0.0,
			"max_abs_az_no_g": 0.0,
			"max_abs_gz": 0.0,
		}
		_preprocess_cache = (0, 0.0, 0.0, result)
		return result

	t_series: List[float] = []
	az_series: List[float] = []
	az_no_g_series: List[float] = []
	gz_series: List[float] = []

	for s in buffer:
		t = float(s.get("t", 0.0))
		az = float(s.get("az", 0.0))
		gz = float(s.get("gz", 0.0))
		t_series.append(t)
		az_series.append(az)
		az_no_g_series.append(az - 9.8)
		gz_series.append(gz)

	count = len(t_series)
	duration = t_series[-1] - t_series[0] if count > 1 else 0.0
	dt = duration / (count - 1) if count > 1 and duration > 0.0 else 0.0

	# Light smoothing at ~10 Hz to emphasise jump‑scale structure.
	cutoff_hz = 10.0
	if dt > 0.0:
		az_smooth = _smooth_1pole(az_series, dt, cutoff_hz)
		az_no_g_smooth = _smooth_1pole(az_no_g_series, dt, cutoff_hz)
		gz_smooth = _smooth_1pole(gz_series, dt, cutoff_hz)
	else:
		az_smooth = list(az_series)
		az_no_g_smooth = list(az_no_g_series)
		gz_smooth = list(gz_series)

	# Optimize max computation by computing during single pass
	max_abs_az = 0.0
	max_abs_az_no_g = 0.0
	max_abs_gz = 0.0
	for v in az_series:
		abs_v = abs(v)
		if abs_v > max_abs_az:
			max_abs_az = abs_v
	for v in az_no_g_series:
		abs_v = abs(v)
		if abs_v > max_abs_az_no_g:
			max_abs_az_no_g = abs_v
	for v in gz_series:
		abs_v = abs(v)
		if abs_v > max_abs_gz:
			max_abs_gz = abs_v

	result = {
		"t": t_series,
		"az": az_series,
		"az_no_g": az_no_g_series,
		"gz": gz_series,
		"az_smooth": az_smooth,
		"az_no_g_smooth": az_no_g_smooth,
		"gz_smooth": gz_smooth,
		"count": count,
		"duration": duration,
		"max_abs_az": max_abs_az,
		"max_abs_az_no_g": max_abs_az_no_g,
		"max_abs_gz": max_abs_gz,
	}
	
	# Update cache
	_preprocess_cache = (len(buffer), float(buffer[0].get("t", 0.0)), float(buffer[-1].get("t", 0.0)), result)
	
	return result


def find_vertical_peaks(
	preprocessed: Dict[str, Any],
	min_height: float,
	min_distance_samples: int,
) -> List[int]:
	"""
	Phase 2.2 – basic peak finder for the gravity‑removed vertical
	acceleration magnitude (|az_no_g|).

	We avoid bringing in external dependencies (e.g. SciPy) by using a
	simple local‑maxima search with:
	  - |az_no_g[i]| >= min_height
	  - At least `min_distance_samples` between accepted peaks.

	Returns a list of indices into the `preprocessed["az_no_g_smooth"]`
	(or `["az_no_g"]` if smoothing is unavailable) / `["t"]`
	series where peaks were detected.
	"""
	series = preprocessed.get("az_no_g_smooth") or preprocessed.get("az_no_g") or []
	n = len(series)
	if n < 3 or min_distance_samples <= 0:
		return []

	peaks: List[int] = []
	last_idx = -min_distance_samples - 1

	for i in range(1, n - 1):
		val = abs(float(series[i]))
		if val < min_height:
			continue

		# Local maximum in terms of absolute value.
		if val < abs(float(series[i - 1])) or val < abs(float(series[i + 1])):
			continue

		if i - last_idx < min_distance_samples:
			continue

		peaks.append(i)
		last_idx = i

	return peaks


def build_jump_windows(
	preprocessed: Dict[str, Any],
	peak_indices: List[int],
	min_flight_time_s: float,
	max_flight_time_s: float,
) -> List[Dict[str, float]]:
	"""
	Phase 2.3 – build simple candidate jump windows from vertical peaks.

	We treat earlier peaks as potential take‑off events and later peaks as
	potential landing events. For each ordered pair (i, j) with j > i, we
	accept a window if the time difference lies within
	[min_flight_time_s, max_flight_time_s].

	Returns a list of dicts with:
	    {
	        "i_takeoff": int,
	        "i_landing": int,
	        "t_takeoff": float,
	        "t_landing": float,
	        "flight_time": float,
	    }
	No scoring or de‑duplication is done here; the realtime wrapper will
	handle higher‑level logic and debouncing in later steps.
	"""
	t_series = preprocessed.get("t") or []
	n_t = len(t_series)
	if n_t == 0 or len(peak_indices) < 2:
		return []

	min_T = max(0.0, float(min_flight_time_s))
	max_T = max(min_T, float(max_flight_time_s))

	windows: List[Dict[str, float]] = []

	for idx_i in range(len(peak_indices)):
		i = peak_indices[idx_i]
		if i < 0 or i >= n_t:
			continue
		t_i = float(t_series[i])

		for idx_j in range(idx_i + 1, len(peak_indices)):
			j = peak_indices[idx_j]
			if j < 0 or j >= n_t:
				continue
			t_j = float(t_series[j])

			T_f = t_j - t_i
			if T_f < min_T or T_f > max_T:
				continue

			windows.append(
				{
					"i_takeoff": int(i),
					"i_landing": int(j),
					"t_takeoff": t_i,
					"t_landing": t_j,
					"flight_time": T_f,
				}
			)

	return windows


def compute_window_metrics(
	preprocessed: Dict[str, Any],
	windows: List[Dict[str, float]],
	g_m_s2: float = 9.8,
) -> List[Dict[str, float]]:
	"""
	Phase 2.4 – enrich candidate windows with simple jump metrics.

	For each window we compute:
	  - `height` from flight time using h = g * T_f^2 / 8.
	  - `peak_az_no_g`: max |a_z - g| within the window.
	  - `peak_gz`: max |ω_z| within the window.
	  - `t_peak_gz`: time at which |ω_z| is maximal in the window
	        (approximate rotation-peak timing).
	"""
	if not windows:
		return []

	t_series = preprocessed.get("t") or []
	az_no_g = (
		preprocessed.get("az_no_g_smooth")
		or preprocessed.get("az_no_g")
		or []
	)
	gz = preprocessed.get("gz_smooth") or preprocessed.get("gz") or []
	n = min(len(t_series), len(az_no_g), len(gz))
	if n == 0:
		return []

	out: List[Dict[str, float]] = []

	# Refinement search window around coarse peaks (seconds). We use sign changes
	# of az_no_g_smooth to estimate liftoff/landing contact more precisely:
	# - Takeoff: first crossing from >=0 to <0 AFTER the takeoff impulse peak.
	# - Landing: last crossing from <0 to >=0 BEFORE the landing impulse peak.
	# This aligns with az_no_g: positive during push-off/impact (above g), and
	# negative during flight (accelerometer reads ~0 so az_no_g ~ -g).
	REFINE_WINDOW_S = 0.25

	# Estimate dt (best-effort) to convert seconds -> sample counts.
	if n >= 2:
		duration_all = float(t_series[-1]) - float(t_series[0])
		dt_est = duration_all / float(max(1, n - 1)) if duration_all > 0.0 else 0.0
	else:
		dt_est = 0.0
	refine_samples = int(REFINE_WINDOW_S / dt_est) if dt_est > 0.0 else int(REFINE_WINDOW_S * 100)
	refine_samples = max(4, min(refine_samples, 200))  # keep bounded

	for w in windows:
		i0 = int(w.get("i_takeoff", 0))
		i1 = int(w.get("i_landing", 0))
		if i0 < 0 or i1 >= n or i1 <= i0:
			continue

		seg_az = az_no_g[i0 : i1 + 1]
		seg_gz = gz[i0 : i1 + 1]
		if not seg_az or not seg_gz:
			continue

		# Compute max values and track peak index in single pass
		peak_az_no_g = 0.0
		peak_gz = 0.0
		peak_gz_rel_idx = 0
		for v in seg_az:
			abs_v = abs(float(v))
			if abs_v > peak_az_no_g:
				peak_az_no_g = abs_v
		for idx, v in enumerate(seg_gz):
			abs_v = abs(float(v))
			if abs_v > peak_gz:
				peak_gz = abs_v
				peak_gz_rel_idx = idx
		peak_gz_idx = i0 + peak_gz_rel_idx
		t_peak_gz = float(t_series[peak_gz_idx]) if peak_gz_idx < n else float(
			w.get("t_takeoff", 0.0)
		)

		# --- Refine takeoff/landing times around the coarse peak indices ---
		t_takeoff_peak = float(w.get("t_takeoff", t_series[i0]))
		t_landing_peak = float(w.get("t_landing", t_series[i1]))

		# Takeoff refinement: search forward for first pos->neg crossing
		t_takeoff_ref = _find_crossing_time(
			t=t_series,
			v=az_no_g,
			start_idx=i0,
			end_idx=min(n - 1, i0 + refine_samples),
			crossing="pos_to_neg",
		)
		if t_takeoff_ref is None:
			t_takeoff_ref = t_takeoff_peak

		# Landing refinement: search backward for last neg->pos crossing before landing peak.
		landing_start = max(0, i1 - refine_samples)
		t_landing_ref: Optional[float] = None
		# Find the last neg->pos crossing in [landing_start, i1] by scanning forward.
		for k in range(landing_start, i1):
			k2 = k + 1
			if k2 >= n:
				break
			v0 = float(az_no_g[k])
			v1 = float(az_no_g[k2])
			if v0 < 0.0 and v1 >= 0.0:
				t_landing_ref = _interp_time_at_zero(float(t_series[k]), v0, float(t_series[k2]), v1)
		if t_landing_ref is None:
			t_landing_ref = t_landing_peak

		# Ensure ordering; if refinement produced an invalid window, fall back to peak times.
		if not (t_landing_ref > t_takeoff_ref):
			t_takeoff_ref = t_takeoff_peak
			t_landing_ref = t_landing_peak

		# --- Step 2.2: revolution counting + under-rotation estimate ---
		# gz_smooth is in degrees/second in our pipeline (Movesense gyro units).
		# We bias-correct using a pre-takeoff baseline window, then integrate to angle.
		BIAS_START_S = 0.5  # seconds before takeoff
		BIAS_END_S = 0.1    # seconds before takeoff

		bias_t0 = t_takeoff_ref - BIAS_START_S
		bias_t1 = t_takeoff_ref - BIAS_END_S
		gz_bias_samples: List[float] = []
		if bias_t1 > bias_t0:
			for i in range(n):
				ti = float(t_series[i])
				if ti < bias_t0:
					continue
				if ti > bias_t1:
					break
				gz_bias_samples.append(float(gz[i]))
		gz_bias = _median(gz_bias_samples)

		# Build bias-corrected gz over the full series (lightweight; n is small ~300)
		gz_corr_deg_s = [float(v) - float(gz_bias) for v in gz]

		# Find integration bounds (indices) for refined takeoff/landing times
		i_takeoff_ref = 0
		while i_takeoff_ref < n and float(t_series[i_takeoff_ref]) < t_takeoff_ref:
			i_takeoff_ref += 1
		i_landing_ref = i_takeoff_ref
		while i_landing_ref < n and float(t_series[i_landing_ref]) < t_landing_ref:
			i_landing_ref += 1
		i_takeoff_ref = max(0, min(i_takeoff_ref, n - 1))
		i_landing_ref = max(0, min(i_landing_ref, n - 1))

		# Integrate gz_corr in radians/second to angle (radians)
		deg_to_rad = math.pi / 180.0
		gz_corr_rad_s = [v * deg_to_rad for v in gz_corr_deg_s]
		theta_z_rad = _integrate_trapz(t_series, gz_corr_rad_s, i_takeoff_ref, i_landing_ref)

		# Revolutions are magnitude of rotation about vertical axis
		revolutions_est = abs(theta_z_rad) / (2.0 * math.pi) if math.pi != 0.0 else 0.0
		revolutions_class = int(round(revolutions_est))
		underrotation = float(revolutions_class) - float(revolutions_est)
		underrot_flag = bool(underrotation < -0.25)

		T_f = float(t_landing_ref - t_takeoff_ref)
		if T_f < 0.0:
			continue
		height = g_m_s2 * (T_f ** 2) / 8.0

		enriched = dict(w)
		enriched.update(
			{
				"height": float(height),
				"t_takeoff_peak": float(t_takeoff_peak),
				"t_landing_peak": float(t_landing_peak),
				# Refined times overwrite the coarse peak-based times for downstream consumers.
				"t_takeoff": float(t_takeoff_ref),
				"t_landing": float(t_landing_ref),
				"flight_time": float(T_f),
				"gz_bias": float(gz_bias),
				"theta_z_rad": float(theta_z_rad),
				"revolutions_est": float(revolutions_est),
				"revolutions_class": int(revolutions_class),
				"underrotation": float(underrotation),
				"underrot_flag": bool(underrot_flag),
				"peak_az_no_g": float(peak_az_no_g),
				"peak_gz": float(peak_gz),
				"t_peak_gz": t_peak_gz,
			}
		)
		out.append(enriched)

	return out


def select_jump_events(
	windows_with_metrics: List[Dict[str, float]],
	min_height_m: float,
	min_peak_az_no_g: float,
	min_peak_gz_deg_s: float,
	min_separation_s: float,
	last_t_takeoff_persistent: Optional[float] = None,
) -> Tuple[List[Dict[str, float]], float]:
	"""
	Phase 2.5 – filter and score enriched windows into jump events.

	Filtering:
	  - Require minimum height, vertical impulse (peak_az_no_g) and
	    rotation speed (peak_gz).
	  - Enforce a minimum separation in time between emitted jumps.

	Scoring (heuristic confidence in [0, 1]):
	  - Normalize height, peak_az_no_g and peak_gz against nominal
	    "good jump" scales and combine with a simple rotation‑phase term.
	  - Very low‑confidence events (< ~0.35) are discarded.

	Returns:
	  - Tuple of (selected_events, updated_last_t_takeoff)
	  - updated_last_t_takeoff can be used to persist separation tracking across analysis cycles
	"""
	if not windows_with_metrics:
		return ([], last_t_takeoff_persistent if last_t_takeoff_persistent is not None else -1e9)

	# Filter by hard thresholds.
	candidates: List[Dict[str, float]] = []
	for w in windows_with_metrics:
		h = float(w.get("height", 0.0))
		a = float(w.get("peak_az_no_g", 0.0))
		g = float(w.get("peak_gz", 0.0))
		if h < min_height_m or a < min_peak_az_no_g or g < min_peak_gz_deg_s:
			continue
		candidates.append(w)

	if not candidates:
		return ([], last_t_takeoff_persistent if last_t_takeoff_persistent is not None else -1e9)

	# Sort by takeoff time.
	candidates.sort(key=lambda x: float(x.get("t_takeoff", 0.0)))

	selected: List[Dict[str, float]] = []
	# Use takeoff-to-takeoff separation (per Bruening et al. 2018), not rotation peak time.
	# This prevents adjacent jumps from being emitted when rotation peaks are close.
	# Initialize with persistent value if provided, otherwise start fresh for this batch.
	last_t_takeoff: float = last_t_takeoff_persistent if last_t_takeoff_persistent is not None else -1e9
	min_sep = max(0.0, float(min_separation_s))

	for w in candidates:
		T_f = float(w.get("flight_time", 0.0))
		h = float(w.get("height", 0.0))
		a = float(w.get("peak_az_no_g", 0.0))
		g = float(w.get("peak_gz", 0.0))
		t_takeoff = float(w.get("t_takeoff", 0.0))
		t_landing = float(w.get("t_landing", t_takeoff + T_f))
		t_peak = float(w.get("t_peak_gz", t_takeoff))

		# Separation check: use takeoff time (per Bruening paper methodology)
		if t_takeoff - last_t_takeoff < min_sep:
			continue

		# Compute a confidence score based on rough nominal scales and
		# how "centrally" the rotation peak lies within the flight.
		def _clip01(x: float) -> float:
			return max(0.0, min(1.0, x))

		norm_h = _clip01(h / 0.6)  # ~0.6 m as a "strong" multi‑rev jump
		norm_a = _clip01(a / 6.0)  # ~6 m/s² vertical impulse above g
		norm_g = _clip01(g / 800.0)  # ~800°/s as nominal strong rotation

		T_f_safe = T_f if T_f > 1e-6 else 1.0
		phase = (t_peak - float(w.get("t_takeoff", 0.0))) / T_f_safe
		# Simple bell‑shaped preference around ~0.6 of flight.
		phase_term = 1.0 - min(abs(phase - 0.6) / 0.5, 1.0)

		conf_raw = 0.35 * norm_h + 0.25 * norm_a + 0.25 * norm_g + 0.15 * phase_term
		conf = _clip01(conf_raw)

		# Discard very low‑confidence windows outright.
		if conf < 0.35:
			continue

		# Overlap-based suppression: if this jump overlaps with a previously selected jump,
		# keep only the one with higher confidence (prevents double-counting from landing spikes).
		overlaps_with = None
		for i, sel in enumerate(selected):
			sel_takeoff = float(sel.get("t_takeoff", 0.0))
			sel_landing = float(sel.get("t_landing", sel_takeoff + float(sel.get("flight_time", 0.0))))
			# Check if windows overlap (with small tolerance for edge cases)
			if not (t_landing < sel_takeoff - 0.1 or t_takeoff > sel_landing + 0.1):
				overlaps_with = i
				break

		if overlaps_with is not None:
			# Found overlap: compare confidence and keep the higher one
			sel = selected[overlaps_with]
			sel_conf = float(sel.get("confidence", 0.0))
			if conf > sel_conf:
				# Replace lower-confidence jump with this one
				event = dict(w)
				event["confidence"] = float(conf)
				event["t_peak"] = t_peak
				event["rotation_phase"] = float(phase)
				selected[overlaps_with] = event
				# Update last_t_takeoff if this jump is later
				if t_takeoff > last_t_takeoff:
					last_t_takeoff = t_takeoff
			# else: keep existing higher-confidence jump, skip this one
			continue

		event = dict(w)
		event["confidence"] = float(conf)
		event["t_peak"] = t_peak
		event["rotation_phase"] = float(phase)

		selected.append(event)
		last_t_takeoff = t_takeoff

	# Return both selected events and the updated last_t_takeoff for persistence
	return (selected, last_t_takeoff)


