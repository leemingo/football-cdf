"""HTTP client and path resolution for SkillCorner raw match bundles.

Single source of truth for the SkillCorner API access patterns previously
duplicated across ``download_skillcorner_kleague.py`` and the
``notebooks/get_skillcorner.ipynb`` exploration notebook. Used by both the
batch downloader CLI and ``SkillcornerDataPreprocessor`` 's auto-download
fallback.

On-disk layout uses ``match_id`` as the primary key:
``{root}/{match_id}/{match_meta.json|tracking.jsonl|dynamic_events.csv}``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from requests.auth import HTTPBasicAuth


BASE_URL = "https://skillcorner.com/api"

# Local filename -> API endpoint suffix relative to ``/match/{match_id}/``.
MATCH_FILES: dict[str, str] = {
    "match_meta.json": "",
    "tracking.jsonl": "tracking/",
    "dynamic_events.csv": "dynamic_events/",
}

# Read-side compat: older notebook downloads wrote ``match.json``.
# All new downloads emit ``match_meta.json`` (the first entry).
META_FILENAMES: tuple[str, ...] = ("match_meta.json", "match.json")


@dataclass(frozen=True)
class SkillcornerCredentials:
    username: str
    password: str

    def to_basic_auth(self) -> HTTPBasicAuth:
        return HTTPBasicAuth(self.username, self.password)

    @classmethod
    def from_env(cls) -> "SkillcornerCredentials":
        username = os.environ.get("SKILLCORNER_USERNAME")
        password = os.environ.get("SKILLCORNER_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "SkillCorner credentials not found. Set SKILLCORNER_USERNAME and "
                "SKILLCORNER_PASSWORD environment variables, or pass an explicit "
                "SkillcornerCredentials instance."
            )
        return cls(username=username, password=password)


def _has_metadata_bundle(path: Path) -> bool:
    return any((path / name).exists() for name in META_FILENAMES)


def find_match_dir(root: str | os.PathLike[str], match_id: str | int) -> Path | None:
    """Return the on-disk match directory for ``match_id`` under ``root``.

    Tries the flat ``{root}/{match_id}/`` layout first, then falls back to the
    legacy season-nested layout (``{root}/{season}/{date}_{home}_vs_{away}_{match_id}/``)
    for compatibility with previously downloaded data. Returns ``None`` when no
    bundle is found.
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


class SkillcornerClient:
    """Thin SkillCorner REST client with pagination + per-match download."""

    def __init__(
        self,
        credentials: SkillcornerCredentials | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.credentials = credentials or SkillcornerCredentials.from_env()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.auth = self.credentials.to_basic_auth()

    # HTTP -----------------------------------------------------------------

    def get(self, endpoint: str, *, params: dict | None = None) -> requests.Response:
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    def iter_paginated(
        self, endpoint: str, *, params: dict | None = None
    ) -> Iterator[dict]:
        url, query = endpoint, params
        while url:
            payload = self.get(url, params=query).json()
            yield from payload.get("results", [])
            url, query = payload.get("next"), None

    # Catalog --------------------------------------------------------------

    def list_competitions(self) -> list[dict]:
        return list(self.iter_paginated("/competitions"))

    def list_seasons(self, competition_id: int) -> list[dict]:
        return list(
            self.iter_paginated("/seasons", params={"competition": competition_id})
        )

    def list_matches(self, season_id: int) -> list[dict]:
        return list(self.iter_paginated("/matches", params={"season": season_id}))

    # Download -------------------------------------------------------------

    def download_match(
        self,
        match_id: str | int,
        root: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Download the three artifacts for ``match_id`` into ``{root}/{match_id}/``.

        Idempotent: existing non-empty files are kept unless ``overwrite=True``.
        """
        match_id_str = str(match_id)
        match_dir = Path(root) / match_id_str
        match_dir.mkdir(parents=True, exist_ok=True)

        for filename, suffix in MATCH_FILES.items():
            target = match_dir / filename
            if not overwrite and target.exists() and target.stat().st_size > 0:
                continue

            response = self.get(f"/match/{match_id_str}/{suffix}")
            if filename.endswith(".json"):
                target.write_text(
                    json.dumps(response.json(), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                target.write_bytes(response.content)

        return match_dir

    def find_or_download_match(
        self,
        match_id: str | int,
        root: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Reuse an existing bundle under ``root`` if found (any layout), else
        download into the flat ``{root}/{match_id}/`` layout."""
        if not overwrite:
            existing = find_match_dir(root, match_id)
            if existing is not None:
                return existing
        Path(root).mkdir(parents=True, exist_ok=True)
        return self.download_match(match_id=match_id, root=root, overwrite=overwrite)
