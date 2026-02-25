"""
P2Skating application modules package.

This file exists to ensure `modules.*` imports work reliably across environments.
"""

from pathlib import Path


def _read_version() -> str:
	try:
		vf = Path(__file__).resolve().parents[1] / "VERSION"
		if vf.exists():
			val = vf.read_text(encoding="utf-8").strip()
			if val:
				return val
	except Exception:
		pass
	return "0.1.0"


__version__ = _read_version()


