from __future__ import annotations

from typing import Optional

from modules.pose.base import PoseProvider
from modules.pose.types import Keypoint, PoseFrame


COCO17_NAMES = [
	"nose",
	"left_eye",
	"right_eye",
	"left_ear",
	"right_ear",
	"left_shoulder",
	"right_shoulder",
	"left_elbow",
	"right_elbow",
	"left_wrist",
	"right_wrist",
	"left_hip",
	"right_hip",
	"left_knee",
	"right_knee",
	"left_ankle",
	"right_ankle",
]


class MediaPipePoseProvider(PoseProvider):
	"""
	MediaPipe Pose provider that outputs a canonical COCO-17-ish keypoint set.

	Notes:
	- MediaPipe uses normalized coordinates; we convert to pixel space.
	- `visibility` is used as score (best-effort).
	"""

	def __init__(
		self,
		model_complexity: int = 1,
		min_detection_confidence: float = 0.5,
		min_tracking_confidence: float = 0.5,
	) -> None:
		try:
			import mediapipe as mp  # type: ignore
		except Exception as e:
			raise RuntimeError(
				"MediaPipe is not installed. Install pose deps with: pip install -r requirements_pose.txt"
			) from e

		self._mp = mp
		self._pose = mp.solutions.pose.Pose(
			static_image_mode=False,
			model_complexity=int(model_complexity),
			enable_segmentation=False,
			smooth_landmarks=True,
			min_detection_confidence=float(min_detection_confidence),
			min_tracking_confidence=float(min_tracking_confidence),
		)

	def name(self) -> str:
		return "mediapipe_pose"

	def infer_rgb(self, rgb, t_video: Optional[float] = None) -> PoseFrame:
		# rgb: HxWx3
		h, w = int(rgb.shape[0]), int(rgb.shape[1])
		res = self._pose.process(rgb)
		out = PoseFrame(backend=self.name(), width=w, height=h, t_video=t_video)
		if not res or not getattr(res, "pose_landmarks", None):
			return out

		lm = res.pose_landmarks.landmark
		# Map COCO-ish names using MediaPipe PoseLandmark indices
		PL = self._mp.solutions.pose.PoseLandmark
		mapping = {
			"nose": PL.NOSE,
			"left_eye": PL.LEFT_EYE,
			"right_eye": PL.RIGHT_EYE,
			"left_ear": PL.LEFT_EAR,
			"right_ear": PL.RIGHT_EAR,
			"left_shoulder": PL.LEFT_SHOULDER,
			"right_shoulder": PL.RIGHT_SHOULDER,
			"left_elbow": PL.LEFT_ELBOW,
			"right_elbow": PL.RIGHT_ELBOW,
			"left_wrist": PL.LEFT_WRIST,
			"right_wrist": PL.RIGHT_WRIST,
			"left_hip": PL.LEFT_HIP,
			"right_hip": PL.RIGHT_HIP,
			"left_knee": PL.LEFT_KNEE,
			"right_knee": PL.RIGHT_KNEE,
			"left_ankle": PL.LEFT_ANKLE,
			"right_ankle": PL.RIGHT_ANKLE,
		}
		for name, idx in mapping.items():
			try:
				p = lm[int(idx)]
				out.keypoints[name] = Keypoint(
					name=name,
					x_px=float(p.x) * float(w),
					y_px=float(p.y) * float(h),
					score=float(getattr(p, "visibility", 0.0) or 0.0),
				)
			except Exception:
				continue
		return out

	def close(self) -> None:
		try:
			if self._pose:
				self._pose.close()
		except Exception:
			pass


