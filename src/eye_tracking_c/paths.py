from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Package lives in src/eye_tracking_c; repo root is parents[2]."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(rel: str) -> Path:
    return repo_root() / rel
