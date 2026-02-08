from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from modules.pose.types import PoseFrame


def _angle_unwrap(series: List[float]) -> List[float]:
	"""
	Unwrap angles in radians so integration doesn't jump at +/-pi.
	"""
	if not series:
		return []
	out = [series[0]]
	for a in series[1:]:
		prev = out[-1]
		d = a - prev
		while d > math.pi:
			a -= 2.0 * math.pi
			d = a - prev
		while d < -math.pi:
			a += 2.0 * math.pi
			d = a - prev
		out.append(a)
	return out


def estimate_revolutions_from_shoulders(frames: List[PoseFrame], min_score: float = 0.35) -> Optional[float]:
	"""
	Very simple 2D proxy: integrate the angle of the shoulder line in the image plane.
	This is NOT a true 3D spin measurement, but it can be useful as a prototype feature.
	"""
	angles: List[float] = []
	for f in frames:
		ls = f.get("left_shoulder")
		rs = f.get("right_shoulder")
		if not ls or not rs:
			continue
		if float(ls.score) < min_score or float(rs.score) < min_score:
			continue
		dx = float(rs.x_px) - float(ls.x_px)
		dy = float(rs.y_px) - float(ls.y_px)
		angles.append(math.atan2(dy, dx))
	if len(angles) < 3:
		return None
	uu = _angle_unwrap(angles)
	delta = float(uu[-1] - uu[0])
	return abs(delta) / (2.0 * math.pi) if math.pi else None


def height_from_flight_time(flight_time_s: float) -> float:
	"""
	Projectile model: h = g * T^2 / 8 (symmetric takeoff/landing).
	"""
	g = 9.80665
	T = max(0.0, float(flight_time_s))
	return g * (T ** 2) / 8.0


def summarize_pose_run(frames: List[PoseFrame]) -> Dict[str, Any]:
	"""
	Small summary for storing in jumps.pose_meta.
	"""
	return {
		"backend": (frames[0].backend if frames else None),
		"frames": len(frames),
		"t0": (frames[0].t_video if frames else None),
		"t1": (frames[-1].t_video if frames else None),
	}


