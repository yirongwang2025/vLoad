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
from typing import Dict, Any, List


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
	"""
	if not buffer:
		return {
			"t": [],
			"az": [],
			"az_no_g": [],
			"gz": [],
			"count": 0,
			"duration": 0.0,
			"max_abs_az": 0.0,
			"max_abs_az_no_g": 0.0,
			"max_abs_gz": 0.0,
		}

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

	max_abs_az = max(abs(v) for v in az_series)
	max_abs_az_no_g = max(abs(v) for v in az_no_g_series)
	max_abs_gz = max(abs(v) for v in gz_series)

	return {
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

	for w in windows:
		i0 = int(w.get("i_takeoff", 0))
		i1 = int(w.get("i_landing", 0))
		if i0 < 0 or i1 >= n or i1 <= i0:
			continue

		seg_az = az_no_g[i0 : i1 + 1]
		seg_gz = gz[i0 : i1 + 1]
		if not seg_az or not seg_gz:
			continue

		abs_az = [abs(float(v)) for v in seg_az]
		abs_gz = [abs(float(v)) for v in seg_gz]

		peak_az_no_g = max(abs_az)
		peak_gz = max(abs_gz)

		peak_gz_rel_idx = abs_gz.index(peak_gz)
		peak_gz_idx = i0 + peak_gz_rel_idx
		t_peak_gz = float(t_series[peak_gz_idx]) if peak_gz_idx < n else float(
			w.get("t_takeoff", 0.0)
		)

		T_f = float(w.get("flight_time", 0.0))
		if T_f < 0.0:
			continue
		height = g_m_s2 * (T_f ** 2) / 8.0

		enriched = dict(w)
		enriched.update(
			{
				"height": float(height),
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
) -> List[Dict[str, float]]:
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
	"""
	if not windows_with_metrics:
		return []

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
		return []

	# Sort by takeoff time.
	candidates.sort(key=lambda x: float(x.get("t_takeoff", 0.0)))

	selected: List[Dict[str, float]] = []
	last_t_peak: float = -1e9
	min_sep = max(0.0, float(min_separation_s))

	for w in candidates:
		T_f = float(w.get("flight_time", 0.0))
		h = float(w.get("height", 0.0))
		a = float(w.get("peak_az_no_g", 0.0))
		g = float(w.get("peak_gz", 0.0))
		t_peak = float(w.get("t_peak_gz", w.get("t_takeoff", 0.0)))

		if t_peak - last_t_peak < min_sep:
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

		event = dict(w)
		event["confidence"] = float(conf)
		event["t_peak"] = t_peak
		event["rotation_phase"] = float(phase)

		selected.append(event)
		last_t_peak = t_peak

	return selected


