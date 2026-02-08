from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from modules.pose.types import PoseFrame


class PoseProvider(ABC):
	"""
	Model adapter interface.

	Implementations should take an RGB image (H,W,3 uint8) and return a PoseFrame.
	"""

	@abstractmethod
	def name(self) -> str: ...

	@abstractmethod
	def infer_rgb(self, rgb, t_video: Optional[float] = None) -> PoseFrame: ...

	@abstractmethod
	def close(self) -> None: ...


