"""
Phase 2.1 – basic preprocessing of vertical acceleration / gyro.

This module provides lightweight utilities to convert the rolling buffer
maintained by `JumpDetectorRealtime` into simple arrays and summary
statistics. It does NOT perform peak finding or jump detection yet.
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


