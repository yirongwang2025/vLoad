"""Shared row→dict mapping helpers for modules.db. Reduces repetition of 30+ field logic in list/get."""
from datetime import date, datetime
from typing import Any, Dict, Optional

Record = Any  # asyncpg Record or dict-like


def _timestamp(val: Any) -> Optional[float]:
	"""Convert datetime to epoch seconds, or None."""
	if val is None:
		return None
	try:
		if hasattr(val, "timestamp"):
			return float(val.timestamp())
		return float(val)
	except (TypeError, ValueError):
		return None


def _iso(val: Any) -> Optional[str]:
	"""Convert datetime/date to ISO string, or None."""
	if val is None:
		return None
	try:
		if hasattr(val, "isoformat"):
			return val.isoformat()
		return str(val)
	except (TypeError, ValueError):
		return None


def _opt_int(row: Record, key: str) -> Optional[int]:
	v = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
	if v is None:
		return None
	try:
		return int(v)
	except (TypeError, ValueError):
		return None


def _opt_float(row: Record, key: str) -> Optional[float]:
	v = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
	if v is None:
		return None
	try:
		return float(v)
	except (TypeError, ValueError):
		return None


def _opt_bool(row: Record, key: str) -> Optional[bool]:
	v = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
	if v is None:
		return None
	return bool(v)


def _opt_str(row: Record, key: str) -> Optional[str]:
	v = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
	if v is None:
		return None
	return str(v) if v else None


def device_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map a devices row to a dict for API responses."""
	return {
		"id": _opt_int(row, "id") or 0,
		"mac_address": str(row["mac_address"]) if row.get("mac_address") else "",
		"name": str(row["name"]) if row.get("name") else "",
		"created_at": _iso(row.get("created_at")),
		"updated_at": _iso(row.get("updated_at")),
	}


def coach_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map a coaches row to a dict for API responses."""
	return {
		"id": _opt_int(row, "id") or 0,
		"name": str(row["name"]) if row.get("name") else "",
		"email": _opt_str(row, "email"),
		"phone": _opt_str(row, "phone"),
		"certification": _opt_str(row, "certification"),
		"level": _opt_str(row, "level"),
		"club": _opt_str(row, "club"),
		"notes": _opt_str(row, "notes"),
		"created_at": _iso(row.get("created_at")),
		"updated_at": _iso(row.get("updated_at")),
	}


def skater_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map a skaters row to a dict for API responses."""
	return {
		"id": _opt_int(row, "id") or 0,
		"name": str(row["name"]) if row.get("name") else "",
		"date_of_birth": _iso(row.get("date_of_birth")),
		"gender": _opt_str(row, "gender"),
		"level": _opt_str(row, "level"),
		"club": _opt_str(row, "club"),
		"email": _opt_str(row, "email"),
		"phone": _opt_str(row, "phone"),
		"notes": _opt_str(row, "notes"),
		"created_at": _iso(row.get("created_at")),
		"updated_at": _iso(row.get("updated_at")),
	}


def jump_row_to_dict(row: Record, include_extra: bool = True) -> Dict[str, Any]:
	"""
	Map a jumps row to a dict for API responses.
	include_extra: if True, includes gz_bias, pose_meta and other fields from get_jump_with_imu.
	"""
	d: Dict[str, Any] = {
		"jump_id": _opt_int(row, "id") or _opt_int(row, "jump_id") or 0,
		"event_id": row.get("event_id"),
		"session_id": row.get("session_id"),
		"video_path": row.get("video_path"),
		"t_peak": _timestamp(row.get("t_peak")),
		"t_start": _timestamp(row.get("t_start")),
		"t_end": _timestamp(row.get("t_end")),
		"t_takeoff_calc": _timestamp(row.get("t_takeoff_calc")),
		"t_landing_calc": _timestamp(row.get("t_landing_calc")),
		"t_takeoff_video": _timestamp(row.get("t_takeoff_video")),
		"t_takeoff_video_t": _opt_float(row, "t_takeoff_video_t"),
		"t_landing_video": _timestamp(row.get("t_landing_video")),
		"t_landing_video_t": _opt_float(row, "t_landing_video_t"),
		"theta_z_rad": _opt_float(row, "theta_z_rad"),
		"revolutions_est": _opt_float(row, "revolutions_est"),
		"revolutions_class": _opt_int(row, "revolutions_class"),
		"underrotation": _opt_float(row, "underrotation"),
		"underrot_flag": _opt_bool(row, "underrot_flag"),
		"flight_time_marked": _opt_float(row, "flight_time_marked"),
		"height_marked": _opt_float(row, "height_marked"),
		"rotation_phase_marked": _opt_float(row, "rotation_phase_marked"),
		"theta_z_rad_marked": _opt_float(row, "theta_z_rad_marked"),
		"revolutions_est_marked": _opt_float(row, "revolutions_est_marked"),
		"revolutions_class_marked": _opt_int(row, "revolutions_class_marked"),
		"underrotation_marked": _opt_float(row, "underrotation_marked"),
		"underrot_flag_marked": _opt_bool(row, "underrot_flag_marked"),
		"flight_time_pose": _opt_float(row, "flight_time_pose"),
		"height_pose": _opt_float(row, "height_pose"),
		"revolutions_pose": _opt_float(row, "revolutions_pose"),
		"flight_time": row.get("flight_time"),
		"height": row.get("height"),
		"acc_peak": row.get("acc_peak"),
		"gyro_peak": row.get("gyro_peak"),
		"rotation_phase": row.get("rotation_phase"),
		"confidence": row.get("confidence"),
		"name": row.get("name"),
		"note": row.get("note"),
		"created_at": _timestamp(row.get("created_at")),
	}
	if include_extra:
		d["gz_bias"] = _opt_float(row, "gz_bias")
		d["gz_bias_marked"] = _opt_float(row, "gz_bias_marked")
		pm = row.get("pose_meta")
		d["pose_meta"] = dict(pm) if pm is not None else None
	return d


def frame_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map a jump_frames row to a dict for API responses."""
	return {
		"frame_idx": int(row["frame_idx"]) if row.get("frame_idx") is not None else 0,
		"t_video": float(row["t_video"]) if row.get("t_video") is not None else 0.0,
		"t_host": float(row["t_host"]) if row.get("t_host") is not None else 0.0,
		"device_ts": _opt_float(row, "device_ts"),
		"width": _opt_int(row, "width"),
		"height": _opt_int(row, "height"),
	}


def imu_sample_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map an imu_samples row to a dict with t, imu_timestamp, acc, gyro, mag."""
	return {
		"t": _timestamp(row.get("t")),
		"imu_timestamp": row.get("imu_timestamp"),
		"acc": [row.get("acc_x"), row.get("acc_y"), row.get("acc_z")],
		"gyro": [row.get("gyro_x"), row.get("gyro_y"), row.get("gyro_z")],
		"mag": [row.get("mag_x"), row.get("mag_y"), row.get("mag_z")],
	}


def skater_coach_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map skater_coaches join row (skater_id, skater_name) – for get_coach_skaters."""
	return {
		"id": _opt_int(row, "id") or 0,
		"skater_id": _opt_int(row, "skater_id") or 0,
		"skater_name": str(row.get("skater_name") or ""),
		"is_head_coach": bool(row.get("is_head_coach", False)),
	}


def coach_skater_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map skater_coaches join row (coach_id, coach_name) – for get_skater_coaches."""
	return {
		"id": _opt_int(row, "id") or 0,
		"coach_id": _opt_int(row, "coach_id") or 0,
		"coach_name": str(row.get("coach_name") or ""),
		"is_head_coach": bool(row.get("is_head_coach", False)),
	}


def skater_device_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map skater_devices join row for get_skater_devices."""
	return {
		"id": _opt_int(row, "id") or 0,
		"device_id": _opt_int(row, "device_id") or 0,
		"device_name": str(row.get("device_name") or ""),
		"mac_address": str(row.get("mac_address") or ""),
		"placement": _opt_str(row, "placement") or "waist",
	}


def skater_coach_link_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map skater_coaches raw row (RETURNING from insert/update) for add_skater_coach."""
	return {
		"id": _opt_int(row, "id") or 0,
		"skater_id": _opt_int(row, "skater_id") or 0,
		"coach_id": _opt_int(row, "coach_id") or 0,
		"is_head_coach": bool(row.get("is_head_coach", False)),
	}


def skater_device_link_row_to_dict(row: Record) -> Dict[str, Any]:
	"""Map skater_devices raw row (RETURNING from insert/update) for add_skater_device."""
	return {
		"id": _opt_int(row, "id") or 0,
		"skater_id": _opt_int(row, "skater_id") or 0,
		"device_id": _opt_int(row, "device_id") or 0,
		"placement": str(row.get("placement") or "waist"),
	}
