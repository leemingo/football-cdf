"""Public path helpers for SkillCorner match bundles.

The helpers in this module only resolve local files. They do not perform API
requests or handle credentials.
"""
from __future__ import annotations

import os
from pathlib import Path


# Read-side compat: older notebook downloads wrote ``match.json``.
# All new downloads emit ``match_meta.json`` when normalizing bundles.
META_FILENAMES: tuple[str, ...] = ("match_meta.json", "match.json")


def _has_metadata_bundle(path: Path) -> bool:
    return any((path / name).exists() for name in META_FILENAMES)


def find_match_dir(root: str | os.PathLike[str], match_id: str | int) -> Path | None:
    """Return the on-disk match directory for ``match_id`` under ``root``.

    Tries the flat ``{root}/{match_id}/`` layout first, then falls back to a
    season-nested layout such as
    ``{root}/{season}/{date}_{home}_vs_{away}_{match_id}/``. Returns ``None``
    when no local bundle is found.
    """
    root_path = Path(root)
    if not root_path.exists():
        return None

    match_id_str = str(match_id)

    direct = root_path / match_id_str
    if direct.is_dir() and _has_metadata_bundle(direct):
        return direct

    if _has_metadata_bundle(root_path) and (
        root_path.name == match_id_str or root_path.name.endswith(f"_{match_id_str}")
    ):
        return root_path

    for candidate in sorted(root_path.rglob(match_id_str)):
        if candidate.is_dir() and _has_metadata_bundle(candidate):
            return candidate

    for candidate in sorted(root_path.rglob(f"*_{match_id_str}")):
        if candidate.is_dir() and _has_metadata_bundle(candidate):
            return candidate

    return None
