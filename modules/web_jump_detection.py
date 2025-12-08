"""
Phase 2.x – utilities for vertical acceleration / gyro processing.

This module currently provides:
  - Phase 2.1: basic preprocessing of vertical series.
  - Phase 2.2: simple peak finding on the gravity‑removed vertical signal.
  - Phase 2.3: utilities for pairing peaks into candidate jump windows and
    computing simple flight‑time information.
"""

from typing import Dict, Any, List


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

	max_abs_az = max(abs(v) for v in az_series)
	max_abs_az_no_g = max(abs(v) for v in az_no_g_series)
	max_abs_gz = max(abs(v) for v in gz_series)

	return {
		"t": t_series,
		"az": az_series,
		"az_no_g": az_no_g_series,
		"gz": gz_series,
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

	Returns a list of indices into the `preprocessed["az_no_g"]` / `["t"]`
	series where peaks were detected.
	"""
	series = preprocessed.get("az_no_g") or []
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
					"i_takeoff": float(i),
					"i_landing": float(j),
					"t_takeoff": t_i,
					"t_landing": t_j,
					"flight_time": T_f,
				}
			)

	return windows


