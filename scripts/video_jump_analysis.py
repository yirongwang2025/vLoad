from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

@dataclass
class FrameObs:
	frame_idx: int
	t_video: float
	foot_y_px: Optional[float]
	body_mid_y_px: Optional[float]
	valid_pixels: int


def _parse_roi(s: str) -> Optional[Tuple[int, int, int, int]]:
	txt = (s or "").strip()
	if not txt:
		return None
	parts = [p.strip() for p in txt.split(",")]
	if len(parts) != 4:
		raise ValueError("ROI must be x,y,w,h")
	x, y, w, h = [int(float(v)) for v in parts]
	if w <= 0 or h <= 0:
		raise ValueError("ROI width/height must be > 0")
	return (x, y, w, h)


def _default_output_prefix() -> Path:
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	return Path("data") / "analysis" / f"video_jump_analysis_{ts}"


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
	if win <= 1:
		return x.copy()
	pad = win // 2
	xp = np.pad(x, (pad, pad), mode="edge")
	k = np.ones(win, dtype=np.float64) / float(win)
	return np.convolve(xp, k, mode="valid")


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
	y = x.copy()
	n = len(y)
	mask = np.isfinite(y)
	if np.all(mask):
		return y
	if not np.any(mask):
		return y
	idx = np.arange(n, dtype=np.float64)
	y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
	return y


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
	segments: List[Tuple[int, int]] = []
	n = len(mask)
	i = 0
	while i < n:
		if not bool(mask[i]):
			i += 1
			continue
		j = i
		while j + 1 < n and bool(mask[j + 1]):
			j += 1
		segments.append((i, j))
		i = j + 1
	return segments


def _probe_video(video_path: Path) -> Dict[str, Any]:
	cmd = [
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"v:0",
		"-show_entries",
		"stream=width,height,r_frame_rate",
		"-show_entries",
		"format=duration",
		"-of",
		"json",
		str(video_path),
	]
	p = subprocess.run(cmd, capture_output=True, text=True, check=False)
	if p.returncode != 0:
		raise RuntimeError(f"ffprobe failed: {p.stderr.strip()[:300]}")
	try:
		info = json.loads(p.stdout)
		stream = (info.get("streams") or [])[0]
		width = int(stream.get("width"))
		height = int(stream.get("height"))
		rf = str(stream.get("r_frame_rate") or "30/1")
		num, den = rf.split("/", 1)
		fps = float(num) / max(1e-9, float(den))
		duration = float((info.get("format") or {}).get("duration") or 0.0)
	except Exception as e:
		raise RuntimeError(f"Failed parsing ffprobe output: {e!r}") from e
	return {"width": width, "height": height, "fps": fps, "duration": duration}


def _iter_gray_frames(video_path: Path, target_fps: float, width: int, height: int):
	frame_bytes = width * height
	cmd = [
		"ffmpeg",
		"-v",
		"error",
		"-i",
		str(video_path),
		"-vf",
		f"fps={float(target_fps):.6f},format=gray",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"gray",
		"-",
	]
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	assert p.stdout is not None
	try:
		idx = 0
		while True:
			buf = p.stdout.read(frame_bytes)
			if not buf or len(buf) < frame_bytes:
				break
			frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width))
			yield idx, frame
			idx += 1
	finally:
		try:
			p.stdout.close()
		except Exception:
			pass
		try:
			p.kill()
		except Exception:
			pass


def _pick_best_segment(
	segments: List[Tuple[int, int]],
	y_smooth: np.ndarray,
	baseline_y: float,
	t: np.ndarray,
	min_airtime_s: float,
	max_airtime_s: float,
) -> Optional[Tuple[int, int, int, float]]:
	best: Optional[Tuple[int, int, int, float]] = None
	clip_mid = 0.5 * float(t[-1] + t[0]) if len(t) > 0 else 0.0
	target_mid = min(2.0, max(0.5, clip_mid * 0.9))
	target_air = min(0.7, max(0.25, 0.35 * float(t[-1] - t[0]) if len(t) > 1 else 0.5))
	for s, e in segments:
		if e <= s:
			continue
		apex_local = int(np.argmin(y_smooth[s : e + 1]))
		apex = s + apex_local
		lift = float(baseline_y - y_smooth[apex])
		dur = float(t[e] - t[s])
		if dur <= 0.0:
			continue
		mid = 0.5 * float(t[s] + t[e])
		time_bias = 1.0 / (1.0 + abs(mid - target_mid))
		dur_bias = 1.0 / (1.0 + abs(dur - target_air) / max(0.05, target_air))
		in_range_bonus = 1.6 if (dur >= min_airtime_s and dur <= max_airtime_s) else 0.8
		score = lift * dur * time_bias * dur_bias * in_range_bonus
		if best is None or score > best[3]:
			best = (s, e, apex, score)
	return best


def _extract_motion_series(
	video_path: Path,
	max_fps: float,
	motion_percentile: float,
) -> Tuple[List[FrameObs], Dict[str, Any]]:
	info = _probe_video(video_path)
	width = int(info["width"])
	height = int(info["height"])
	src_fps = float(info["fps"])
	duration = float(info["duration"])
	target_fps = min(max(1.0, float(max_fps)), max(1.0, src_fps))

	out: List[FrameObs] = []
	x_track = width // 2
	roi_half_w = max(20, int(0.18 * width))
	y0 = int(0.18 * height)
	y1 = int(0.98 * height)
	bg: Optional[np.ndarray] = None
	alpha = 0.995

	for i, gray in _iter_gray_frames(video_path, target_fps=target_fps, width=width, height=height):
		t_video = float(i) / float(target_fps)
		gf = gray.astype(np.float32)
		if bg is None:
			bg = gf.copy()
			out.append(FrameObs(frame_idx=i, t_video=t_video, foot_y_px=None, body_mid_y_px=None, valid_pixels=0))
			continue

		diff = np.abs(gf - bg)
		bg = alpha * bg + (1.0 - alpha) * gf

		xl = max(0, x_track - roi_half_w)
		xr = min(width, x_track + roi_half_w)
		if xr <= xl:
			xl, xr = 0, width

		roi = diff[y0:y1, xl:xr]
		if roi.size == 0:
			out.append(FrameObs(frame_idx=i, t_video=t_video, foot_y_px=None, body_mid_y_px=None, valid_pixels=0))
			continue

		thr = float(np.percentile(roi, max(50.0, min(99.9, float(motion_percentile)))))
		thr = max(6.0, thr)
		mask = roi >= thr
		valid_pixels = int(mask.sum())
		if valid_pixels < 40:
			out.append(FrameObs(frame_idx=i, t_video=t_video, foot_y_px=None, body_mid_y_px=None, valid_pixels=valid_pixels))
			continue

		ys, xs = np.where(mask)
		xs_full = xs + xl
		ys_full = ys + y0
		x_track = int(np.median(xs_full))
		foot_y = float(np.percentile(ys_full, 96))
		body_mid_y = float(np.percentile(ys_full, 60))
		out.append(
			FrameObs(
				frame_idx=i,
				t_video=t_video,
				foot_y_px=foot_y,
				body_mid_y_px=body_mid_y,
				valid_pixels=valid_pixels,
			)
		)

	meta = {
		"fps": src_fps,
		"duration_s": duration,
		"width": width,
		"height": height,
		"effective_fps": target_fps,
		"roi_half_width_px": roi_half_w,
		"roi_y0_px": y0,
		"roi_y1_px": y1,
	}
	return out, meta


def _extract_motion_series_cv2(
	video_path: Path,
	max_fps: float,
	roi_xywh: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[FrameObs], Dict[str, Any]]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

	src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
	if src_fps <= 0.0:
		src_fps = 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	target_fps = min(max(1.0, float(max_fps)), max(1.0, src_fps))
	stride = max(1, int(round(src_fps / target_fps)))

	backsub = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=24, detectShadows=False)
	kernel = np.ones((3, 3), np.uint8)
	min_area = max(80, int(0.0006 * max(1, width * height)))
	if roi_xywh is not None:
		rx, ry, rw, rh = roi_xywh
		x0 = max(0, min(width - 1, rx))
		y0 = max(0, min(height - 1, ry))
		x1 = max(x0 + 1, min(width, rx + rw))
		y1 = max(y0 + 1, min(height, ry + rh))
	else:
		# Default ROI: center-heavy lower area (where skater body is likely visible).
		x0 = int(0.22 * width)
		x1 = int(0.78 * width)
		y0 = int(0.18 * height)
		y1 = int(0.98 * height)
	roi_area = max(1, (x1 - x0) * (y1 - y0))

	out: List[FrameObs] = []
	prev_cx: Optional[float] = None
	prev_cy: Optional[float] = None
	i = 0
	try:
		while True:
			ok, frame = cap.read()
			if not ok or frame is None:
				break
			if i % stride != 0:
				i += 1
				continue
			t_video = float(i) / float(src_fps)

			fg = backsub.apply(frame)
			fg = cv2.medianBlur(fg, 5)
			_, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
			fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
			fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
			roi_mask = fg[y0:y1, x0:x1]

			num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
			best_idx = -1
			best_score = -1e9
			diag = math.hypot(float(x1 - x0), float(y1 - y0))
			for k in range(1, int(num_labels)):
				area = int(stats[k, cv2.CC_STAT_AREA])
				if area < min_area:
					continue
				x = int(stats[k, cv2.CC_STAT_LEFT])
				y = int(stats[k, cv2.CC_STAT_TOP])
				w = int(stats[k, cv2.CC_STAT_WIDTH])
				h = int(stats[k, cv2.CC_STAT_HEIGHT])
				cx = float(x + 0.5 * w + x0)
				cy = float(y + 0.5 * h + y0)
				area_norm = min(1.0, float(area) / float(max(200.0, 0.08 * roi_area)))
				if prev_cx is not None and prev_cy is not None:
					d = math.hypot(cx - prev_cx, cy - prev_cy)
					cont = math.exp(-2.5 * d / max(1.0, diag))
				else:
					cont = 0.5
				center_x = 1.0 - min(1.0, abs(cx - (0.5 * (x0 + x1))) / max(1.0, 0.5 * (x1 - x0)))
				lower = min(1.0, max(0.0, (cy - y0) / max(1.0, (y1 - y0))))
				aspect = float(h) / max(1.0, float(w))
				aspect_bonus = min(1.0, max(0.0, (aspect - 0.5) / 2.5))
				score = 1.8 * cont + 1.1 * area_norm + 0.6 * lower + 0.5 * center_x + 0.4 * aspect_bonus
				if score > best_score:
					best_score = score
					best_idx = k

			if best_idx < 0:
				out.append(FrameObs(frame_idx=i, t_video=t_video, foot_y_px=None, body_mid_y_px=None, valid_pixels=0))
				i += 1
				continue

			x = int(stats[best_idx, cv2.CC_STAT_LEFT])
			y = int(stats[best_idx, cv2.CC_STAT_TOP])
			w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
			h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
			comp_mask = labels[y : y + h, x : x + w] == best_idx
			yy, xx = np.where(comp_mask)
			if yy.size < 20:
				out.append(FrameObs(frame_idx=i, t_video=t_video, foot_y_px=None, body_mid_y_px=None, valid_pixels=int(yy.size)))
				i += 1
				continue
			yy_full = yy + y + y0
			xx_full = xx + x + x0
			prev_cx = float(np.median(xx_full))
			prev_cy = float(np.median(yy_full))
			foot_y = float(np.percentile(yy_full, 96))
			body_mid_y = float(np.percentile(yy_full, 60))
			out.append(
				FrameObs(
					frame_idx=i,
					t_video=t_video,
					foot_y_px=foot_y,
					body_mid_y_px=body_mid_y,
					valid_pixels=int(yy.size),
				)
			)
			i += 1
	finally:
		try:
			cap.release()
		except Exception:
			pass

	meta = {
		"fps": src_fps,
		"duration_s": (float(frame_count) / float(src_fps)) if src_fps > 0.0 and frame_count > 0 else None,
		"width": width,
		"height": height,
		"effective_fps": float(src_fps) / float(stride),
		"frame_count": frame_count,
		"stride": stride,
		"min_component_area_px": min_area,
		"roi": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
	}
	return out, meta


def _detect_jump(
	obs: List[FrameObs],
	video_height: int,
	lift_threshold_px: Optional[float],
	min_airtime_s: float,
	max_airtime_s: float,
) -> Dict[str, Any]:
	if len(obs) < 10:
		return {"ok": False, "error": "not enough frames"}

	t = np.array([o.t_video for o in obs], dtype=np.float64)
	foot = np.array(
		[float("nan") if o.foot_y_px is None else float(o.foot_y_px) for o in obs],
		dtype=np.float64,
	)
	valid_mask = np.isfinite(foot)
	valid_ratio = float(np.mean(valid_mask))
	if valid_ratio < 0.5:
		return {"ok": False, "error": f"insufficient visible ankles (valid_ratio={valid_ratio:.2f})"}

	foot_interp = _interp_nan_1d(foot)
	dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30.0
	dt = max(1e-4, dt)
	win = max(3, int(round(0.12 / dt)))
	if win % 2 == 0:
		win += 1
	foot_smooth = _moving_average(foot_interp, win=win)

	n = len(foot_smooth)
	edge_n = max(5, int(round(0.2 * n)))
	baseline_y = float(np.median(np.concatenate([foot_smooth[:edge_n], foot_smooth[-edge_n:]])))

	auto_lift = max(6.0, 0.015 * float(video_height if video_height > 0 else 720))
	lift_px = float(lift_threshold_px) if lift_threshold_px is not None else float(auto_lift)
	lift_px = max(2.0, lift_px)

	air_mask = foot_smooth <= (baseline_y - lift_px)
	segments = _segments_from_mask(air_mask)
	best = _pick_best_segment(
		segments=segments,
		y_smooth=foot_smooth,
		baseline_y=baseline_y,
		t=t,
		min_airtime_s=float(min_airtime_s),
		max_airtime_s=float(max_airtime_s),
	)
	if best is None:
		return {
			"ok": False,
			"error": "no airborne segment found",
			"valid_ratio": valid_ratio,
			"baseline_y_px": baseline_y,
			"lift_threshold_px": lift_px,
		}

	s, e, apex, score = best
	t_takeoff = float(t[s])
	t_landing = float(t[e])
	t_apex = float(t[apex])
	airtime = float(t_landing - t_takeoff)
	peak_lift = float(baseline_y - foot_smooth[apex])

	if airtime < float(min_airtime_s) or airtime > float(max_airtime_s):
		return {
			"ok": False,
			"error": f"airtime out of range: {airtime:.3f}s",
			"t_takeoff_s": t_takeoff,
			"t_landing_s": t_landing,
			"t_apex_s": t_apex,
			"airtime_s": airtime,
			"peak_lift_px": peak_lift,
			"valid_ratio": valid_ratio,
			"baseline_y_px": baseline_y,
			"lift_threshold_px": lift_px,
		}

	# Heuristic confidence in [0,1], based on visibility + lift + duration plausibility.
	conf_vis = min(1.0, max(0.0, valid_ratio))
	conf_lift = min(1.0, max(0.0, peak_lift / max(1.0, 2.0 * lift_px)))
	conf_time = 1.0
	if airtime < 0.2:
		conf_time = max(0.0, airtime / 0.2)
	elif airtime > 1.0:
		conf_time = max(0.0, (1.5 - airtime) / 0.5)
	conf = float(max(0.0, min(1.0, 0.45 * conf_vis + 0.4 * conf_lift + 0.15 * conf_time)))

	return {
		"ok": True,
		"t_takeoff_s": t_takeoff,
		"t_landing_s": t_landing,
		"t_apex_s": t_apex,
		"airtime_s": airtime,
		"peak_lift_px": peak_lift,
		"baseline_y_px": baseline_y,
		"lift_threshold_px": lift_px,
		"segment_start_idx": int(s),
		"segment_end_idx": int(e),
		"apex_idx": int(apex),
		"segment_score": float(score),
		"valid_ratio": valid_ratio,
		"confidence": conf,
	}


def _write_timeseries_csv(path: Path, obs: List[FrameObs]) -> None:
	fieldnames = ["frame_idx", "t_video", "foot_y_px", "hip_y_px", "valid_points"]
	with path.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.DictWriter(fh, fieldnames=fieldnames)
		writer.writeheader()
		for o in obs:
			writer.writerow(
				{
					"frame_idx": o.frame_idx,
					"t_video": o.t_video,
					"foot_y_px": o.foot_y_px,
					"hip_y_px": o.body_mid_y_px,
					"valid_points": o.valid_pixels,
				}
			)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Independent video-only jump analysis from pose landmarks.")
	parser.add_argument(
		"--video",
		default="data/sessions/20260228_113342_detect/jump_clips/jump_15.mp4",
		help="Path to jump clip video",
	)
	parser.add_argument(
		"--output-prefix",
		default=str(_default_output_prefix()),
		help="Prefix path for outputs (writes <prefix>_timeseries.csv and <prefix>_summary.json)",
	)
	parser.add_argument("--max-fps", type=float, default=30.0, help="Max FPS to process")
	parser.add_argument(
		"--method",
		default="ffmpeg_motion",
		choices=["ffmpeg_motion", "cv2_motion"],
		help="Detection backend method",
	)
	parser.add_argument(
		"--cv2-roi",
		default="",
		help="Optional ROI for cv2 method as x,y,w,h (pixels)",
	)
	parser.add_argument(
		"--motion-percentile",
		type=float,
		default=98.0,
		help="Foreground motion percentile threshold (higher=more selective)",
	)
	parser.add_argument(
		"--lift-threshold-px",
		type=float,
		default=None,
		help="Fixed airborne lift threshold in pixels (default: auto by video height)",
	)
	parser.add_argument("--min-airtime-s", type=float, default=0.15, help="Minimum plausible airtime")
	parser.add_argument("--max-airtime-s", type=float, default=1.2, help="Maximum plausible airtime")
	args = parser.parse_args(argv)

	video_path = Path(args.video).expanduser().resolve()
	if not video_path.exists():
		print(f"[video-jump] Video not found: {video_path}")
		return 2

	prefix = Path(args.output_prefix).expanduser().resolve()
	prefix.parent.mkdir(parents=True, exist_ok=True)
	timeseries_path = prefix.with_name(prefix.name + "_timeseries.csv")
	summary_path = prefix.with_name(prefix.name + "_summary.json")

	try:
		if args.method == "cv2_motion":
			roi = _parse_roi(str(args.cv2_roi or ""))
			obs, meta = _extract_motion_series_cv2(
				video_path=video_path,
				max_fps=float(args.max_fps),
				roi_xywh=roi,
			)
		else:
			obs, meta = _extract_motion_series(
				video_path=video_path,
				max_fps=float(args.max_fps),
				motion_percentile=float(args.motion_percentile),
			)
		det = _detect_jump(
			obs=obs,
			video_height=int(meta.get("height") or 0),
			lift_threshold_px=args.lift_threshold_px,
			min_airtime_s=float(args.min_airtime_s),
			max_airtime_s=float(args.max_airtime_s),
		)

		_write_timeseries_csv(timeseries_path, obs)
		summary = {
			"video_path": str(video_path),
			"method": str(args.method),
			"meta": meta,
			"detection": det,
			"generated_at": datetime.now().isoformat(),
		}
		summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

		print(f"[video-jump] Timeseries: {timeseries_path}")
		print(f"[video-jump] Summary: {summary_path}")
		if det.get("ok"):
			print(
				"[video-jump] Jump: "
				f"takeoff={det['t_takeoff_s']:.3f}s, "
				f"apex={det['t_apex_s']:.3f}s, "
				f"landing={det['t_landing_s']:.3f}s, "
				f"airtime={det['airtime_s']:.3f}s, "
				f"confidence={det['confidence']:.2f}"
			)
			return 0
		print(f"[video-jump] Detection failed: {det.get('error')}")
		return 1
	except Exception as e:
		print(f"[video-jump] Fatal error: {e!r}")
		return 2


if __name__ == "__main__":
	raise SystemExit(main())

