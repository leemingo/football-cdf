"""Local path discovery for the StatsBomb Open Data repository layout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class StatsbombMatchPaths:
    """Resolved source files for one StatsBomb match."""

    data_root: Path
    competitions_path: Path
    matches_path: Path
    events_path: Path
    lineups_path: Path
    three_sixty_path: Path | None


def resolve_statsbomb_data_root(root: str | Path) -> Path:
    """Return the ``data`` directory from a repository root or data root."""
    root_path = Path(root).expanduser()
    candidates = (root_path, root_path / "data")
    for candidate in candidates:
        if (
            (candidate / "competitions.json").is_file()
            and (candidate / "matches").is_dir()
            and (candidate / "events").is_dir()
            and (candidate / "lineups").is_dir()
        ):
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a StatsBomb Open Data directory containing "
        f"competitions.json, matches/, events/, and lineups/. Tried: {tried}"
    )


def find_match_record(
    root: str | Path,
    match_id: str | int,
) -> tuple[Path, dict]:
    """Find the season match file and record for ``match_id``."""
    data_root = resolve_statsbomb_data_root(root)
    wanted = str(match_id)
    record = _match_catalog(str(data_root)).get(wanted)
    if record is not None:
        return record
    raise FileNotFoundError(
        f"Could not find StatsBomb match_id={wanted!r} below {data_root / 'matches'}"
    )


@lru_cache(maxsize=8)
def _match_catalog(data_root: str) -> dict[str, tuple[Path, dict]]:
    """Build a process-local match index so batch conversion scans metadata once."""
    root = Path(data_root)
    catalog: dict[str, tuple[Path, dict]] = {}
    for path in sorted((root / "matches").glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            match_id = record.get("match_id")
            if match_id is not None:
                catalog[str(match_id)] = (path, record)
    return catalog


def resolve_statsbomb_match_paths(
    root: str | Path,
    match_id: str | int,
) -> StatsbombMatchPaths:
    """Resolve all required and optional files for one match."""
    data_root = resolve_statsbomb_data_root(root)
    wanted = str(match_id)
    matches_path, _ = find_match_record(data_root, wanted)
    events_path = data_root / "events" / f"{wanted}.json"
    lineups_path = data_root / "lineups" / f"{wanted}.json"
    missing = [str(path) for path in (events_path, lineups_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required StatsBomb match files are missing: " + ", ".join(missing)
        )

    three_sixty = data_root / "three-sixty" / f"{wanted}.json"
    return StatsbombMatchPaths(
        data_root=data_root,
        competitions_path=data_root / "competitions.json",
        matches_path=matches_path,
        events_path=events_path,
        lineups_path=lineups_path,
        three_sixty_path=three_sixty if three_sixty.is_file() else None,
    )
