"""Hudl flattened StatsBomb event export -> CDF/SPADL preprocessing.

Hudl's export used by this project contains StatsBomb-compatible event
semantics, but stores nested fields as dotted keys (for example
``pass.end_location`` and ``team.id``).  This adapter normalizes one local
event file in memory and delegates the CDF/SPADL conversion to the existing
``StatsbombDataPreprocessor`` implementation.

The source event dictionaries are retained under ``_hudl_flat_event`` in the
normalized raw event so that this adapter does not silently discard provider
fields or invent missing values.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseEventTrackingPreprocessor
from .statsbomb_preprocessing import StatsbombDataPreprocessor


class HudlStatsbombDataPreprocessor(StatsbombDataPreprocessor):
    """Process one local Hudl flattened StatsBomb event JSON file.

    Unlike :class:`StatsbombDataPreprocessor`, this class accepts the path to
    one event file directly.  It does not require the official StatsBomb
    ``matches/``, ``events/``, and ``lineups/`` directory layout.
    """

    SOURCE_PITCH_X = StatsbombDataPreprocessor.SOURCE_PITCH_X
    SOURCE_PITCH_Y = StatsbombDataPreprocessor.SOURCE_PITCH_Y

    _CORE_KEYS = (
        "id",
        "index",
        "match_id",
        "period",
        "timestamp",
        "minute",
        "second",
        "duration",
        "location",
        "related_events",
        "possession",
        "under_pressure",
        "counterpress",
        "off_camera",
        "out",
    )

    _NESTED_OBJECTS = (
        "type",
        "team",
        "player",
        "position",
        "possession_team",
        "play_pattern",
    )

    _EVENT_COMPONENTS = (
        "pass",
        "carry",
        "shot",
        "goalkeeper",
        "duel",
        "foul_committed",
        "foul_won",
        "block",
        "ball_receipt",
        "dribble",
        "interception",
        "50_50",
        "substitution",
        "ball_recovery",
        "clearance",
    )

    @staticmethod
    def _present(value: object) -> bool:
        """Return whether a flat value carries source information."""
        if value is None:
            return False
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, (list, tuple)):
            return bool(value)
        try:
            return not bool(pd.isna(value))
        except (TypeError, ValueError):
            return True

    @classmethod
    def _copy_object(cls, flat: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        """Build a nested ``{id, name}``-like object from dotted keys."""
        output: dict[str, Any] = {}
        for suffix in ("id", "name"):
            key = f"{prefix}.{suffix}"
            if key in flat and cls._present(flat[key]):
                output[suffix] = copy.deepcopy(flat[key])
        return output

    @classmethod
    def _copy_component(cls, flat: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        """Copy a dotted event component into a nested StatsBomb object.

        ``*.end_location.x/y`` are intentionally not used to reconstruct an
        endpoint.  The export supplies ``*.end_location`` for events where it
        is semantically present; retaining only that source field avoids
        fabricating a value when the endpoint is unavailable.
        """
        output: dict[str, Any] = {}
        prefix_dot = f"{prefix}."
        for key, value in flat.items():
            if not key.startswith(prefix_dot):
                continue
            suffix = key[len(prefix_dot) :]
            if not cls._present(value):
                continue
            if suffix.startswith("end_location."):
                continue
            parts = suffix.split(".")
            target: dict[str, Any] = output
            for part in parts[:-1]:
                child = target.get(part)
                if not isinstance(child, dict):
                    child = {}
                    target[part] = child
                target = child
            target[parts[-1]] = copy.deepcopy(value)
        return output

    @classmethod
    def _normalize_event(cls, flat_event: Mapping[str, Any]) -> dict[str, Any]:
        """Convert one Hudl flat event to the nested shape expected upstream."""
        flat = dict(flat_event)
        event: dict[str, Any] = {
            "_hudl_flat_event": copy.deepcopy(flat),
        }

        for key in cls._CORE_KEYS:
            if key in flat and cls._present(flat[key]):
                event[key] = copy.deepcopy(flat[key])

        for prefix in cls._NESTED_OBJECTS:
            value = cls._copy_object(flat, prefix)
            if value:
                event[prefix] = value

        for prefix in cls._EVENT_COMPONENTS:
            value = cls._copy_component(flat, prefix)
            if value:
                event[prefix] = value

        # ``tactics.lineup`` is handled separately because its records use
        # the same flattened convention and are consumed as lineup metadata.
        if cls._present(flat.get("tactics.formation")) or cls._present(
            flat.get("tactics.lineup")
        ):
            tactics: dict[str, Any] = {}
            if cls._present(flat.get("tactics.formation")):
                tactics["formation"] = copy.deepcopy(flat["tactics.formation"])
            if cls._present(flat.get("tactics.lineup")):
                tactics["lineup"] = copy.deepcopy(flat["tactics.lineup"])
            event["tactics"] = tactics

        # Keep scalar context used by the flat export and useful when raw
        # events are inspected, while leaving missing values missing.
        context_keys = (
            "home_score",
            "away_score",
            "Score",
            "TotalScore",
            "OpposingScore",
            "GameState",
            "OpposingGameState",
            "WinningTeam",
            "OpposingTeam",
            "OpposingTeam.id",
            "Goal",
            "TimeOfGoal",
            "ElapsedTime",
            "attacking_direction",
            "kick_off",
            "match_date",
            "match_status",
            "data_version",
            "competition_gender",
            "stadium_name",
            "referee_name",
        )
        for key in context_keys:
            if key in flat and cls._present(flat[key]):
                event[key] = copy.deepcopy(flat[key])

        return event

    @staticmethod
    def _load_payload(event_path: str | Path) -> list[dict[str, Any]]:
        path = Path(event_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
            raise ValueError(f"Hudl event file must contain a JSON array of objects: {path}")
        return [dict(row) for row in payload]

    @classmethod
    def load_event_data(cls, event_path: str) -> pd.DataFrame:
        """Load and normalize one flattened Hudl event file."""
        payload = cls._load_payload(event_path)
        normalized = [cls._normalize_event(event) for event in payload]
        return StatsbombDataPreprocessor._events_from_payload(normalized)

    @classmethod
    def _metadata_payload(cls, payload: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def first(key: str) -> Any:
            for event in payload:
                value = event.get(key)
                if cls._present(value):
                    return copy.deepcopy(value)
            return None

        return {
            "competition": {
                "competition_id": first("competition.competition_id"),
                "competition_name": first("competition.competition_name")
                or first("competition_name"),
                "country_name": first("competition.country_name"),
            },
            "season": {
                "season_id": first("season.season_id"),
                "season_name": first("season.season_name"),
            },
            "match_id": first("match_id"),
            "match_date": first("match_date"),
            "kick_off": first("kick_off"),
            "home_score": first("home_score"),
            "away_score": first("away_score"),
            "home_team": {
                "home_team_id": first("home_team.home_team_id"),
                "home_team_name": first("home_team.home_team_name"),
            },
            "away_team": {
                "away_team_id": first("away_team.away_team_id"),
                "away_team_name": first("away_team.away_team_name"),
            },
            "stadium": {"name": first("stadium_name")},
            "referee": {"name": first("referee_name")},
            "metadata": {"data_version": first("data_version")},
            "match_status": first("match_status"),
            "competition_gender": first("competition_gender"),
        }

    @classmethod
    def extract_match_metadata(cls, raw_metadata: Mapping[str, Any]) -> dict[str, Any]:
        metadata = StatsbombDataPreprocessor.extract_match_metadata(dict(raw_metadata))
        metadata["vendor_name"] = "Hudl StatsBomb"
        return metadata

    @classmethod
    def _raw_starting_lineup(cls, payload: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows_by_team: dict[str, dict[str, Any]] = {}
        for event in payload:
            if event.get("type.name") != "Starting XI":
                continue
            team_id = event.get("team.id")
            if not cls._present(team_id) or team_id is None:
                continue
            team_key = str(team_id)
            lineup = event.get("tactics.lineup")
            if not isinstance(lineup, list):
                continue
            players: list[dict[str, Any]] = []
            for player in lineup:
                if not isinstance(player, Mapping):
                    continue
                player_id = player.get("player.id")
                player_name = player.get("player.name")
                position_name = player.get("position.name")
                position_id = player.get("position.id")
                position: dict[str, Any] = {}
                if cls._present(position_name):
                    position["position"] = position_name
                elif cls._present(position_id):
                    position["position"] = position_id
                position["start_reason"] = "Starting XI"
                normalized_player: dict[str, Any] = {
                    "player_id": player_id if cls._present(player_id) else None,
                    "player_name": player_name if cls._present(player_name) else pd.NA,
                    "jersey_number": player.get("jersey_number", pd.NA),
                    "positions": [position],
                }
                players.append(normalized_player)
            rows_by_team[team_key] = {
                "team_id": team_id,
                "team_name": event.get("team.name", pd.NA),
                "lineup": players,
            }

        if len(rows_by_team) < 2:
            raise ValueError(
                "Hudl event file must contain Starting XI tactics.lineup records for both teams"
            )

        # A standalone event export has no official lineups file.  Add players
        # introduced by substitutions so event rows retain stable ``object_id``
        # values.  Their jersey number and playing position are absent here,
        # therefore those values are intentionally left missing rather than
        # inferred from the player they replace.
        for event in payload:
            if event.get("type.name") != "Substitution":
                continue
            team_key = str(event.get("team.id"))
            team = rows_by_team.get(team_key)
            replacement_id = event.get("substitution.replacement.id")
            if team is None or not cls._present(replacement_id):
                continue
            existing_ids = {
                str(player.get("player_id"))
                for player in team["lineup"]
                if cls._present(player.get("player_id"))
            }
            if str(replacement_id) in existing_ids:
                continue
            team["lineup"].append(
                {
                    "player_id": replacement_id,
                    "player_name": event.get("substitution.replacement.name", pd.NA),
                    "jersey_number": pd.NA,
                    "positions": [],
                }
            )
        return list(rows_by_team.values())

    @classmethod
    def _raw_360_frames(cls, payload: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Build StatsBomb-360-compatible frame records from flat event context.

        Hudl supplies event-linked ``freeze_frame`` and ``visible_area`` in the
        same export, rather than in a separate ``three-sixty`` file.  They are
        context snapshots only; this method deliberately does not treat them
        as continuous tracking data.
        """
        frames: list[dict[str, Any]] = []
        for event in payload:
            event_id = event.get("id")
            freeze_frame = event.get("freeze_frame")
            visible_area = event.get("visible_area")
            if not cls._present(event_id):
                continue
            if not isinstance(freeze_frame, list) and not isinstance(visible_area, list):
                continue
            frames.append(
                {
                    "event_uuid": str(event_id),
                    "freeze_frame": copy.deepcopy(freeze_frame)
                    if isinstance(freeze_frame, list)
                    else [],
                    "visible_area": copy.deepcopy(visible_area)
                    if isinstance(visible_area, list)
                    else [],
                }
            )
        return frames

    def __init__(self, event_path: str):
        BaseEventTrackingPreprocessor.__init__(self)
        self.event_path = str(Path(event_path))
        payload = self._load_payload(self.event_path)
        if not payload:
            raise ValueError(f"Hudl event file is empty: {self.event_path}")

        self.raw_metadata = self._metadata_payload(payload)
        self.match_metadata = self.extract_match_metadata(self.raw_metadata)
        self.match_id = str(self.match_metadata.get("match_id"))
        self.raw_lineup = self._raw_starting_lineup(payload)
        self.lineup = self.load_lineup_data(self.raw_lineup, self.match_metadata)
        normalized = [self._normalize_event(event) for event in payload]
        self.events = StatsbombDataPreprocessor._events_from_payload(normalized)
        self.raw_360 = self._raw_360_frames(payload)
        self.three_sixty_path = None
        self.tracking = pd.DataFrame()
        self.tracking_long = pd.DataFrame()
        self.fps = float("nan")
