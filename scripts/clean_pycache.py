#!/usr/bin/env python3
"""Recursively remove all __pycache__ directories within the vLoad project."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # vLoad folder


def main():
    removed = []
    for d in ROOT.rglob("__pycache__"):
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(str(d.relative_to(ROOT)))
    for p in removed:
        print(f"Removed: {p}")
    print(f"Total: {len(removed)} __pycache__ folder(s) removed.")


if __name__ == "__main__":
    main()
