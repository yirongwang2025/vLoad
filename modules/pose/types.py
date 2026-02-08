from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses import field


@dataclass(frozen=True)
class Keypoint:
	"""
	A single 2D keypoint in pixel coordinates.
	"""

	name: str
	x_px: float
	y_px: float
	score: float  # confidence/visibility [0..1] best-effort


@dataclass(frozen=True)
class PoseFrame:
	"""
	Model-agnostic pose output for a single video frame.

	- Coordinates are in pixel space to keep downstream logic consistent.
	- Times are optional; callers can provide t_video (clip-relative seconds) and/or
	  t_host (epoch seconds) if they have a mapping.
	"""

	backend: str
	width: int
	height: int
	t_video: Optional[float] = None
	t_host: Optional[float] = None
	keypoints: Dict[str, Keypoint] = field(default_factory=dict)

	def get(self, name: str) -> Optional[Keypoint]:
		if not self.keypoints:
			return None
		return self.keypoints.get(name)


