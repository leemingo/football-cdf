"""StatsBomb Open Data -> project CDF-aligned tables and SPADL actions.

The implementation mirrors the public methods exposed by the Bepro and Sportec
preprocessors. StatsBomb Open Data has event-linked 360 freeze frames but no
continuous tracking feed, so 360 is emitted as separate frame/object tables and
is never passed through the tracking or kinematics pipeline.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseEventTrackingPreprocessor
from .constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y
from .contracts import (
    EVENT_CDF_CORE_COLUMNS,
    EVENT_SPADL_HELPER_COLUMNS,
    PROJECT_CANONICAL_VERSION,
    SPADL_ACTION_COLUMNS,
    THREE_SIXTY_FRAME_COLUMNS,
    THREE_SIXTY_OBJECT_COLUMNS,
)
from .statsbomb_mappings import (
    derive_cdf_sub_type,
    derive_cdf_type,
    infer_body_part,
    infer_cdf_outcome,
    nested_name,
    normalize_label,
    normalize_position,
)
from .statsbomb_paths import resolve_statsbomb_match_paths


class StatsbombDataPreprocessor(BaseEventTrackingPreprocessor):
    """Load one match from a local StatsBomb Open Data checkout."""

    SOURCE_PITCH_X = 120.0
    SOURCE_PITCH_Y = 80.0
    _SPADL_EXTENDED_ACTIONS = {"ball_recovery", "dispossessed", "shot_block"}

    def __init__(
        self,
        root_dir: str,
        match_id: str,
        load_360: bool = False,
    ):
        super().__init__()
        self.match_id = str(match_id)
        self.paths = resolve_statsbomb_match_paths(root_dir, self.match_id)
        self.meta_path = self.paths.matches_path
        self.event_path = self.paths.events_path
        self.lineup_path = self.paths.lineups_path
        self.three_sixty_path = self.paths.three_sixty_path

        self.raw_metadata = self.load_raw_metadata(str(self.meta_path), self.match_id)
        self.match_metadata = self.extract_match_metadata(self.raw_metadata)
        self.raw_lineup = self.load_raw_lineup(str(self.lineup_path))
        self.lineup = self.load_lineup_data(self.raw_lineup, self.match_metadata)
        self.events = self.load_event_data(str(self.event_path))

        self.raw_360: list[dict] = []
        if load_360 and self.three_sixty_path is not None:
            self.raw_360 = self.load_360_data(str(self.three_sixty_path))

        # StatsBomb Open Data does not contain continuous tracking.
        self.tracking = pd.DataFrame()
        self.tracking_long = pd.DataFrame()
        self.fps = np.nan

    # ------------------------------------------------------------------
    # Metadata and local raw loaders
    # ------------------------------------------------------------------
    @staticmethod
    def load_raw_metadata(meta_path: str, match_id: str | None = None) -> dict:
        with Path(meta_path).open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"StatsBomb matches file must contain a JSON array: {meta_path}")
        if match_id is None and len(records) == 1:
            return records[0]
        wanted = str(match_id)
        for record in records:
            if str(record.get("match_id")) == wanted:
                return record
        raise ValueError(f"match_id={wanted!r} is not present in {meta_path}")

    @staticmethod
    def extract_match_metadata(raw_metadata: dict) -> dict:
        def string_id(value: object) -> object:
            return str(value) if value is not None and pd.notna(value) else pd.NA

        competition = raw_metadata.get("competition") or {}
        season = raw_metadata.get("season") or {}
        home = raw_metadata.get("home_team") or {}
        away = raw_metadata.get("away_team") or {}
        stadium = raw_metadata.get("stadium") or {}
        referee = raw_metadata.get("referee") or {}
        stage = raw_metadata.get("competition_stage") or {}
        source_meta = raw_metadata.get("metadata") or {}

        match_date = raw_metadata.get("match_date")
        kick_off = raw_metadata.get("kick_off")
        kickoff_time = pd.to_datetime(
            f"{match_date} {kick_off}" if match_date and kick_off else None,
            errors="coerce",
        )
        home_score = raw_metadata.get("home_score", pd.NA)
        away_score = raw_metadata.get("away_score", pd.NA)
        final_score = (
            f"{home_score}-{away_score}"
            if pd.notna(home_score) and pd.notna(away_score)
            else pd.NA
        )
        play_direction = {
            period: "leftright" for period in CDF_PERIOD_MAP.values()
        }

        return {
            "competition_id": string_id(competition.get("competition_id")),
            "competition_name": competition.get("competition_name", pd.NA),
            "season_id": string_id(season.get("season_id")),
            "season_name": season.get("season_name", pd.NA),
            "match_id": string_id(raw_metadata.get("match_id")),
            "kickoff_time": kickoff_time,
            # This is the normalized project frame, not the unknown physical
            # side occupied by the home team in the broadcast.
            "play_direction": play_direction,
            "home_team_id": string_id(home.get("home_team_id")),
            "home_team_name": home.get("home_team_name", pd.NA),
            "away_team_id": string_id(away.get("away_team_id")),
            "away_team_name": away.get("away_team_name", pd.NA),
            "stadium_id": string_id(stadium.get("id")),
            "stadium_name": stadium.get("name", pd.NA),
            "pitch_length": PITCH_X,
            "pitch_width": PITCH_Y,
            "final_home_score": home_score,
            "final_away_score": away_score,
            "final_score": final_score,
            "vendor_name": "StatsBomb",
            "vendor_version": source_meta.get("data_version", pd.NA),
            "cdf_version": PROJECT_CANONICAL_VERSION,
            "country_name": competition.get("country_name", pd.NA),
            "competition_stage_id": string_id(stage.get("id")),
            "competition_stage_name": stage.get("name", pd.NA),
            "match_week": raw_metadata.get("match_week", pd.NA),
            "match_status": raw_metadata.get("match_status", pd.NA),
            "match_status_360": raw_metadata.get("match_status_360", pd.NA),
            "referee_id": string_id(referee.get("id")),
            "referee_name": referee.get("name", pd.NA),
            "shot_fidelity_version": source_meta.get("shot_fidelity_version", pd.NA),
            "xy_fidelity_version": source_meta.get("xy_fidelity_version", pd.NA),
            "source_pitch_length": StatsbombDataPreprocessor.SOURCE_PITCH_X,
            "source_pitch_width": StatsbombDataPreprocessor.SOURCE_PITCH_Y,
            "source_orientation": "action_executing_team_left_to_right",
            "canonical_orientation": "static_home_left",
            "kickoff_timezone_known": False,
        }

    @staticmethod
    def load_raw_lineup(lineup_path: str) -> list[dict]:
        with Path(lineup_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"StatsBomb lineup file must contain a JSON array: {lineup_path}")
        return payload

    @staticmethod
    def _starting_position(positions: list[dict]) -> tuple[object, bool]:
        starters = [
            position
            for position in positions
            if position.get("start_reason") == "Starting XI"
        ]
        selected = starters[0] if starters else (positions[0] if positions else {})
        return normalize_position(selected.get("position")), bool(starters)

    @staticmethod
    def _build_object_id(home_away: object, jersey_number: object, player_id: object) -> object:
        if pd.isna(home_away):
            return pd.NA
        if jersey_number is not None and pd.notna(jersey_number):
            try:
                return f"{home_away}_{int(jersey_number)}"
            except (TypeError, ValueError):
                pass
        if player_id is not None and pd.notna(player_id):
            return f"{home_away}_{player_id}"
        return pd.NA

    @staticmethod
    def load_lineup_data(raw_lineup: list[dict], match_metadata: dict) -> pd.DataFrame:
        home_team_id = str(match_metadata.get("home_team_id"))
        away_team_id = str(match_metadata.get("away_team_id"))
        rows = []
        for team in raw_lineup:
            team_id = str(team.get("team_id"))
            if team_id == home_team_id:
                home_away = "home"
            elif team_id == away_team_id:
                home_away = "away"
            else:
                home_away = pd.NA
            for player in team.get("lineup") or []:
                positions = player.get("positions") or []
                playing_position, starting = (
                    StatsbombDataPreprocessor._starting_position(positions)
                )
                player_id = player.get("player_id")
                jersey = player.get("jersey_number", pd.NA)
                country = player.get("country") or {}
                rows.append(
                    {
                        "team_id": team_id,
                        "team_name": team.get("team_name", pd.NA),
                        "home_away": home_away,
                        "player_id": str(player_id) if player_id is not None else pd.NA,
                        "uniform_number": jersey,
                        "object_id": StatsbombDataPreprocessor._build_object_id(
                            home_away, jersey, player_id
                        ),
                        "player_name": player.get("player_name", pd.NA),
                        "playing_position": playing_position,
                        "starting": starting,
                        "player_nickname": player.get("player_nickname", pd.NA),
                        "country_id": country.get("id", pd.NA),
                        "country_name": country.get("name", pd.NA),
                        "position_intervals": positions,
                        "cards": player.get("cards") or [],
                    }
                )

        lineup = pd.DataFrame(rows)
        if lineup.empty:
            return lineup
        lineup["home_away"] = pd.Categorical(
            lineup["home_away"], categories=["home", "away"], ordered=True
        )
        lineup["uniform_number"] = pd.to_numeric(
            lineup["uniform_number"], errors="coerce"
        ).astype("Int64")
        lineup["starting"] = lineup["starting"].fillna(False).astype(bool)
        for column in ("team_id", "player_id", "object_id"):
            lineup[column] = lineup[column].astype("string")
        return lineup.sort_values(
            ["home_away", "starting", "uniform_number", "player_name"],
            ascending=[True, False, True, True],
            kind="mergesort",
            ignore_index=True,
        )

    @staticmethod
    def _extract_end_location(event: Mapping) -> list | None:
        raw_type = nested_name(event.get("type"))
        field = {
            "Pass": "pass",
            "Carry": "carry",
            "Shot": "shot",
            "Goal Keeper": "goalkeeper",
        }.get(raw_type)
        if field is None:
            return None
        value = (event.get(field) or {}).get("end_location")
        return value if isinstance(value, list) and len(value) >= 2 else None

    @staticmethod
    def _events_from_payload(payload: list[dict]) -> pd.DataFrame:
        rows = []
        for event in payload:
            location = event.get("location") or [None, None]
            end_location = StatsbombDataPreprocessor._extract_end_location(event) or [None, None]
            raw_type = nested_name(event.get("type"))
            team = event.get("team") or {}
            player = event.get("player") or {}
            position = event.get("position") or {}
            possession_team = event.get("possession_team") or {}
            play_pattern = event.get("play_pattern") or {}
            pass_data = event.get("pass") or {}
            shot_data = event.get("shot") or {}
            goalkeeper = event.get("goalkeeper") or {}
            duel = event.get("duel") or {}
            foul = event.get("foul_committed") or {}
            block = event.get("block") or {}
            recipient = pass_data.get("recipient") or {}
            rows.append(
                {
                    "event_id": event.get("id"),
                    "event_index": event.get("index"),
                    "period_id": event.get("period"),
                    "source_timestamp": event.get("timestamp"),
                    "minute": event.get("minute"),
                    "second": event.get("second"),
                    "duration": event.get("duration", pd.NA),
                    "raw_event_type": raw_type,
                    "team_id": team.get("id", pd.NA),
                    "team_name": team.get("name", pd.NA),
                    "player_id": player.get("id", pd.NA),
                    "player_name": player.get("name", pd.NA),
                    "position_id": position.get("id", pd.NA),
                    "position_name": position.get("name", pd.NA),
                    "receiver_player_id": recipient.get("id", pd.NA),
                    "receiver_player_name": recipient.get("name", pd.NA),
                    "coordinates_x": location[0],
                    "coordinates_y": location[1],
                    "end_coordinates_x": end_location[0],
                    "end_coordinates_y": end_location[1],
                    "end_coordinates_z": end_location[2] if len(end_location) > 2 else pd.NA,
                    "possession_id": event.get("possession", pd.NA),
                    "possession_team_id": possession_team.get("id", pd.NA),
                    "possession_team_name": possession_team.get("name", pd.NA),
                    "play_pattern": play_pattern.get("name", pd.NA),
                    "under_pressure": event.get("under_pressure", False),
                    "counterpress": event.get("counterpress", False),
                    "off_camera": event.get("off_camera", False),
                    "out": event.get("out", False),
                    "related_event_ids": event.get("related_events"),
                    "pass_type": nested_name(pass_data.get("type")),
                    "pass_outcome": nested_name(pass_data.get("outcome")),
                    "pass_height": nested_name(pass_data.get("height")),
                    "pass_body_part": nested_name(pass_data.get("body_part")),
                    "pass_length": pass_data.get("length", pd.NA),
                    "pass_angle": pass_data.get("angle", pd.NA),
                    "pass_cross": pass_data.get("cross", False),
                    "pass_assisted_shot_id": pass_data.get("assisted_shot_id", pd.NA),
                    "pass_shot_assist": pass_data.get("shot_assist", False),
                    "pass_goal_assist": pass_data.get("goal_assist", False),
                    "shot_type": nested_name(shot_data.get("type")),
                    "shot_outcome": nested_name(shot_data.get("outcome")),
                    "shot_body_part": nested_name(shot_data.get("body_part")),
                    "shot_technique": nested_name(shot_data.get("technique")),
                    "shot_statsbomb_xg": shot_data.get("statsbomb_xg", pd.NA),
                    "has_shot_freeze_frame": bool(shot_data.get("freeze_frame")),
                    "goalkeeper_type": nested_name(goalkeeper.get("type")),
                    "goalkeeper_outcome": nested_name(goalkeeper.get("outcome")),
                    "duel_type": nested_name(duel.get("type")),
                    "duel_outcome": nested_name(duel.get("outcome")),
                    "card_type": nested_name(foul.get("card")),
                    "foul_type": nested_name(foul.get("type")),
                    "block_save_block": block.get("save_block", False),
                    "raw_event": event,
                }
            )
        events = pd.DataFrame(rows)
        if not events.empty:
            events = events.sort_values("event_index", kind="mergesort", ignore_index=True)
        return events

    @staticmethod
    def load_event_data(event_path: str) -> pd.DataFrame:
        with Path(event_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"StatsBomb event file must contain a JSON array: {event_path}")
        return StatsbombDataPreprocessor._events_from_payload(payload)

    @staticmethod
    def load_360_data(path: str) -> list[dict]:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"StatsBomb 360 file must contain a JSON array: {path}")
        return payload

    # ------------------------------------------------------------------
    # Raw events -> project CDF-aligned events
    # ------------------------------------------------------------------
    @staticmethod
    def _match_clock_seconds(values: pd.Series) -> pd.Series:
        return pd.to_timedelta(values, errors="coerce").dt.total_seconds()

    @staticmethod
    def _scale_center_axis(value: object, source_size: float, target_size: float) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return np.nan
        return float(numeric) / float(source_size) * float(target_size) - float(target_size) / 2.0

    @staticmethod
    def _transform_location(
        location: Iterable | None,
        *,
        rotate: bool,
    ) -> tuple[float, float]:
        values = list(location or [])
        raw_x = values[0] if len(values) > 0 else None
        raw_y = values[1] if len(values) > 1 else None
        x = StatsbombDataPreprocessor._scale_center_axis(
            raw_x, StatsbombDataPreprocessor.SOURCE_PITCH_X, PITCH_X
        )
        y = StatsbombDataPreprocessor._scale_center_axis(
            raw_y, StatsbombDataPreprocessor.SOURCE_PITCH_Y, PITCH_Y
        )
        if rotate:
            x, y = -x, -y
        return round(x, 2) if pd.notna(x) else np.nan, round(y, 2) if pd.notna(y) else np.nan

    @staticmethod
    def add_score_columns(
        events: pd.DataFrame,
        *,
        home_team_id: object,
        away_team_id: object,
    ) -> pd.DataFrame:
        """Add the score immediately before each event without double-counting own goals."""
        events = events.copy()
        home_id = str(home_team_id)
        away_id = str(away_team_id)
        home_score = 0
        away_score = 0
        home_scores: list[int] = []
        away_scores: list[int] = []
        for _, row in events.iterrows():
            home_scores.append(home_score)
            away_scores.append(away_score)
            scoring_team = None
            if row.get("type") == "shot" and row.get("outcome_detailed") == "goal":
                scoring_team = str(row.get("team_id"))
            elif row.get("type") == "own_goal_for":
                # StatsBomb also emits a paired Own Goal Against row. Count the
                # beneficiary row only, then preserve the Against row as the
                # player's SPADL owngoal action.
                scoring_team = str(row.get("team_id"))
            if scoring_team == home_id:
                home_score += 1
            elif scoring_team == away_id:
                away_score += 1
        events["home_score"] = pd.Series(home_scores, index=events.index, dtype="int64")
        events["away_score"] = pd.Series(away_scores, index=events.index, dtype="int64")
        events["score"] = pd.Series(
            [f"{home}-{away}" for home, away in zip(home_scores, away_scores)],
            index=events.index,
            dtype="string",
        )
        return events

    @staticmethod
    def convert_raw_events_to_cdf(
        raw_events: pd.DataFrame | list[dict],
        lineup: Optional[pd.DataFrame] = None,
        *,
        match_metadata: Optional[dict] = None,
        three_sixty_event_ids: Optional[set[str]] = None,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
    ) -> pd.DataFrame:
        events = (
            raw_events.copy()
            if isinstance(raw_events, pd.DataFrame)
            else StatsbombDataPreprocessor._events_from_payload(list(raw_events))
        )
        metadata = match_metadata or {}
        home_team_id = str(metadata.get("home_team_id"))
        away_team_id = str(metadata.get("away_team_id"))
        match_id = str(metadata.get("match_id"))

        if events.empty:
            columns = list(EVENT_CDF_CORE_COLUMNS)
            if include_spadl_helpers:
                columns.extend(EVENT_SPADL_HELPER_COLUMNS)
            return pd.DataFrame(columns=columns)

        events["source_coordinates_x"] = pd.to_numeric(events["coordinates_x"], errors="coerce")
        events["source_coordinates_y"] = pd.to_numeric(events["coordinates_y"], errors="coerce")
        events["source_end_coordinates_x"] = pd.to_numeric(
            events["end_coordinates_x"], errors="coerce"
        )
        events["source_end_coordinates_y"] = pd.to_numeric(
            events["end_coordinates_y"], errors="coerce"
        )

        starts = []
        ends = []
        for _, row in events.iterrows():
            rotate = str(row.get("team_id")) == away_team_id
            starts.append(
                StatsbombDataPreprocessor._transform_location(
                    [row.get("coordinates_x"), row.get("coordinates_y")], rotate=rotate
                )
            )
            ends.append(
                StatsbombDataPreprocessor._transform_location(
                    [row.get("end_coordinates_x"), row.get("end_coordinates_y")], rotate=rotate
                )
            )
        events["x"] = [value[0] for value in starts]
        events["y"] = [value[1] for value in starts]
        events["x_end"] = [value[0] for value in ends]
        events["y_end"] = [value[1] for value in ends]

        events["match_id"] = match_id
        events["utc_timestamp"] = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
        events["match_clock"] = events["source_timestamp"].astype("string")
        events["match_clock_seconds"] = StatsbombDataPreprocessor._match_clock_seconds(
            events["source_timestamp"]
        )
        events["period"] = events["period_id"].map(CDF_PERIOD_MAP).astype("string")
        events["type"] = events["raw_event"].apply(derive_cdf_type).astype("string")
        events["sub_type"] = events["raw_event"].apply(derive_cdf_sub_type).astype("string")

        outcomes = events["raw_event"].apply(infer_cdf_outcome)
        events["outcome"] = outcomes.apply(lambda value: value[0]).astype("string")
        events["outcome_detailed"] = outcomes.apply(lambda value: value[1]).astype("string")
        events["success"] = pd.Series(
            [value[2] for value in outcomes], index=events.index, dtype="boolean"
        )
        events["body_part"] = events["raw_event"].apply(infer_body_part).astype("string")
        events["receiver_id"] = events["receiver_player_id"].astype("string")
        events["receiver_time"] = pd.Series(pd.NA, index=events.index, dtype="string")
        events["is_synced"] = pd.Series(False, index=events.index, dtype="boolean")
        events["tracking_frame_id"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
        events["tracking_frame_id_end"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
        events["has_360"] = events["event_id"].astype(str).isin(three_sixty_event_ids or set())

        for column in ("match_id", "event_id", "team_id", "player_id", "receiver_id"):
            events[column] = events[column].astype("string")
        events["event_index"] = pd.to_numeric(
            events["event_index"], errors="coerce"
        ).astype("Int64")

        events["object_id"] = pd.Series(pd.NA, index=events.index, dtype="string")
        events["receiver_object_id"] = pd.Series(pd.NA, index=events.index, dtype="string")
        if lineup is not None and not lineup.empty:
            player_to_object = (
                lineup.dropna(subset=["player_id", "object_id"])
                .drop_duplicates("player_id")
                .set_index("player_id")["object_id"]
                .to_dict()
            )
            events["object_id"] = events["player_id"].map(player_to_object).astype("string")
            events["receiver_object_id"] = events["receiver_id"].map(
                player_to_object
            ).astype("string")

        if include_score:
            events = StatsbombDataPreprocessor.add_score_columns(
                events,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
        else:
            events["home_score"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
            events["away_score"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
            events["score"] = pd.Series(pd.NA, index=events.index, dtype="string")

        raw_helper_columns = [
            "period_id",
            "source_timestamp",
            "minute",
            "second",
            "duration",
            "raw_event_type",
            "team_name",
            "player_name",
            "position_id",
            "position_name",
            "receiver_player_name",
            "possession_id",
            "possession_team_id",
            "possession_team_name",
            "play_pattern",
            "under_pressure",
            "counterpress",
            "off_camera",
            "out",
            "pass_type",
            "pass_outcome",
            "pass_height",
            "pass_body_part",
            "pass_length",
            "pass_angle",
            "pass_cross",
            "pass_assisted_shot_id",
            "pass_shot_assist",
            "pass_goal_assist",
            "shot_type",
            "shot_outcome",
            "shot_body_part",
            "shot_technique",
            "shot_statsbomb_xg",
            "has_shot_freeze_frame",
            "goalkeeper_type",
            "goalkeeper_outcome",
            "duel_type",
            "duel_outcome",
            "card_type",
            "foul_type",
            "block_save_block",
            "has_360",
            "source_coordinates_x",
            "source_coordinates_y",
            "source_end_coordinates_x",
            "source_end_coordinates_y",
            "end_coordinates_z",
        ]

        ordered = [column for column in EVENT_CDF_CORE_COLUMNS if column in events.columns]
        if include_spadl_helpers:
            ordered.extend(
                column for column in EVENT_SPADL_HELPER_COLUMNS if column in events.columns
            )
        if preserve_raw_columns:
            ordered.extend(column for column in raw_helper_columns if column in events.columns)
        return events.loc[:, list(dict.fromkeys(ordered))].reset_index(drop=True)

    @staticmethod
    def to_cdf_events(
        raw_events: pd.DataFrame | list[dict],
        lineup: Optional[pd.DataFrame] = None,
        *,
        match_metadata: Optional[dict] = None,
        three_sixty_event_ids: Optional[set[str]] = None,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
    ) -> pd.DataFrame:
        return StatsbombDataPreprocessor.convert_raw_events_to_cdf(
            raw_events,
            lineup=lineup,
            match_metadata=match_metadata,
            three_sixty_event_ids=three_sixty_event_ids,
            include_score=include_score,
            include_spadl_helpers=include_spadl_helpers,
            preserve_raw_columns=preserve_raw_columns,
        )

    def preprocess_cdf_events(
        self,
        raw_events: Optional[pd.DataFrame] = None,
        *,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
    ) -> pd.DataFrame:
        source = self.events if raw_events is None else raw_events
        event_ids = {str(frame.get("event_uuid")) for frame in self.raw_360}
        return self.to_cdf_events(
            source,
            lineup=self.lineup,
            match_metadata=self.match_metadata,
            three_sixty_event_ids=event_ids,
            include_score=True,
            include_spadl_helpers=include_spadl_helpers,
            preserve_raw_columns=preserve_raw_columns,
        )

    # ------------------------------------------------------------------
    # CDF-aligned events -> SPADL-style actions
    # ------------------------------------------------------------------
    @staticmethod
    def infer_spadl_types(
        events_cdf: pd.DataFrame,
        *,
        include_extended_actions: bool = True,
    ) -> pd.DataFrame:
        events = events_cdf.copy().reset_index(drop=True)
        events["spadl_type"] = pd.Series(pd.NA, index=events.index, dtype="string")

        pass_mask = events["type"] == "pass"
        cross_mask = events["type"] == "cross"
        events.loc[pass_mask, "spadl_type"] = "pass"
        events.loc[cross_mask, "spadl_type"] = "cross"
        events.loc[pass_mask & (events["sub_type"] == "throw_in"), "spadl_type"] = "throw_in"
        events.loc[pass_mask & (events["sub_type"] == "goal_kick"), "spadl_type"] = "goalkick"
        pass_height = events.get(
            "pass_height", pd.Series(pd.NA, index=events.index)
        ).astype("string")
        set_piece_cross = cross_mask | pass_height.eq("High Pass")
        pass_or_cross = pass_mask | cross_mask
        corner = pass_or_cross & (events["sub_type"] == "corner_kick")
        free_kick = pass_or_cross & (events["sub_type"] == "free_kick")
        events.loc[corner & ~set_piece_cross, "spadl_type"] = "corner_short"
        events.loc[corner & set_piece_cross, "spadl_type"] = "corner_crossed"
        events.loc[free_kick & ~set_piece_cross, "spadl_type"] = "freekick_short"
        events.loc[free_kick & set_piece_cross, "spadl_type"] = "freekick_crossed"

        # Keep administrative/unknown pass rows in CDF, but do not present
        # them as analytical SPADL actions.
        pass_non_action = events["outcome_detailed"].isin(
            ["injury_clearance", "unknown"]
        )
        events.loc[pass_or_cross & pass_non_action, "spadl_type"] = pd.NA

        shot_mask = events["type"] == "shot"
        events.loc[shot_mask, "spadl_type"] = "shot"
        events.loc[shot_mask & (events["sub_type"] == "penalty"), "spadl_type"] = "shot_penalty"
        events.loc[shot_mask & (events["sub_type"] == "free_kick"), "spadl_type"] = "shot_freekick"

        direct_mapping = {
            "carry": "dribble",
            "dribble": "take_on",
            "interception": "interception",
            "ball_recovery": "ball_recovery",
            "clearance": "clearance",
            "foul_committed": "foul",
            "miscontrol": "bad_touch",
            "dispossessed": "dispossessed",
            "own_goal_against": "bad_touch",
        }
        for cdf_type, spadl_type in direct_mapping.items():
            events.loc[events["type"] == cdf_type, "spadl_type"] = spadl_type

        duel_type = events.get("duel_type", pd.Series(pd.NA, index=events.index)).astype("string")
        events.loc[
            (events["type"] == "duel") & duel_type.str.contains("Tackle", case=False, na=False),
            "spadl_type",
        ] = "tackle"

        save_block = events.get(
            "block_save_block", pd.Series(False, index=events.index)
        ).fillna(False).astype(bool)
        events.loc[(events["type"] == "block") & save_block, "spadl_type"] = "shot_block"

        keeper_type = events.get(
            "goalkeeper_type", pd.Series(pd.NA, index=events.index)
        ).astype("string").str.lower()
        keeper_outcome = events.get(
            "goalkeeper_outcome", pd.Series(pd.NA, index=events.index)
        ).astype("string").str.lower()
        keeper_mask = events["type"] == "goalkeeper"
        events.loc[
            keeper_mask & keeper_type.str.contains("punch", na=False), "spadl_type"
        ] = "keeper_punch"
        events.loc[
            keeper_mask
            & keeper_type.str.contains("claim|collect|smother", regex=True, na=False),
            "spadl_type",
        ] = "keeper_claim"
        events.loc[
            keeper_mask
            & keeper_type.str.contains("save|shot saved|penalty saved", regex=True, na=False),
            "spadl_type",
        ] = "keeper_save"
        events.loc[
            keeper_mask
            & keeper_type.str.contains("keeper sweeper", na=False)
            & keeper_outcome.str.contains("claim", na=False),
            "spadl_type",
        ] = "keeper_claim"
        events.loc[
            keeper_mask
            & keeper_type.str.contains("keeper sweeper", na=False)
            & ~keeper_outcome.str.contains("claim", na=False),
            "spadl_type",
        ] = "keeper_pick_up"

        if not include_extended_actions:
            events.loc[
                events["spadl_type"].isin(StatsbombDataPreprocessor._SPADL_EXTENDED_ACTIONS),
                "spadl_type",
            ] = pd.NA

        # A clearance has no explicit outcome in StatsBomb. Mirror the existing
        # Sportec rule: it succeeds when the next emitted action in the period
        # belongs to the same team.
        emitted = events[events["spadl_type"].notna()]
        for index in events.index[events["spadl_type"] == "clearance"]:
            following = emitted.loc[index + 1 :]
            if following.empty:
                events.at[index, "success"] = False
            else:
                next_action = following.iloc[0]
                events.at[index, "success"] = bool(
                    next_action.get("period") == events.at[index, "period"]
                    and pd.notna(next_action.get("team_id"))
                    and next_action.get("team_id") == events.at[index, "team_id"]
                )
        events.loc[events["spadl_type"] == "foul", "success"] = False
        events.loc[events["spadl_type"] == "ball_recovery", "success"] = True
        events.loc[events["spadl_type"] == "shot_block", "success"] = True
        events.loc[events["type"] == "own_goal_against", "success"] = False
        return events

    @staticmethod
    def _infer_spadl_bodypart_name(events: pd.DataFrame) -> pd.Series:
        mapping = {
            "foot": "foot",
            "left_foot": "foot_left",
            "right_foot": "foot_right",
            "head": "head",
            "other": "other",
        }
        return events["body_part"].map(mapping).astype("string")

    @staticmethod
    def _infer_spadl_result_name(events: pd.DataFrame) -> pd.Series:
        result = pd.Series(pd.NA, index=events.index, dtype="string")
        result.loc[events["success"] == True] = "success"
        result.loc[events["success"] == False] = "fail"
        result.loc[events["outcome"] == "offside"] = "offside"
        result.loc[events["type"] == "own_goal_against"] = "owngoal"
        cards = events.get("card_type", pd.Series(pd.NA, index=events.index)).astype("string")
        result.loc[cards.str.contains("Yellow", case=False, na=False)] = "yellow_card"
        result.loc[cards.str.contains("Red", case=False, na=False)] = "red_card"
        return result

    @staticmethod
    def _insert_interception_pass_actions(events: pd.DataFrame) -> pd.DataFrame:
        """Split StatsBomb interception passes into interception + pass actions."""
        compound = (
            events["type"].isin(["pass", "cross"])
            & (events["sub_type"] == "interception")
            & events["spadl_type"].notna()
        )
        if not compound.any():
            return events

        interceptions = events.loc[compound].copy()
        interceptions["spadl_type"] = "interception"
        interceptions["success"] = True
        interceptions["receiver_id"] = pd.NA
        if "receiver_object_id" in interceptions.columns:
            interceptions["receiver_object_id"] = pd.NA
        interceptions["body_part"] = "foot"
        interceptions["x_end"] = interceptions["x"]
        interceptions["y_end"] = interceptions["y"]
        interceptions["_spadl_order"] = interceptions["_spadl_order"] - 0.1

        # Drop columns that are entirely missing in the derived rows before
        # concatenation. They are restored by pandas alignment from ``events``
        # and this avoids dtype ambiguity for all-NA extension columns.
        interceptions = interceptions.dropna(axis="columns", how="all")
        combined = pd.concat([events, interceptions], ignore_index=True, sort=False)
        return combined.sort_values("_spadl_order", kind="stable").reset_index(drop=True)

    @staticmethod
    def to_spadl_events(
        events_cdf: pd.DataFrame,
        *,
        infer_end_coordinates: bool = True,
        include_cdf_columns: bool = False,
        include_extended_actions: bool = True,
        split_interception_passes: bool = True,
    ) -> pd.DataFrame:
        events = StatsbombDataPreprocessor.infer_spadl_types(
            events_cdf,
            include_extended_actions=include_extended_actions,
        )
        events["original_event_id"] = events["event_id"].astype("string")
        events["_spadl_order"] = pd.Series(
            range(len(events)), index=events.index, dtype="float64"
        )
        if split_interception_passes:
            events = StatsbombDataPreprocessor._insert_interception_pass_actions(events)
        events["end_x"] = pd.to_numeric(events["x_end"], errors="coerce")
        events["end_y"] = pd.to_numeric(events["y_end"], errors="coerce")
        if infer_end_coordinates:
            events["end_x"] = events["end_x"].fillna(events["x"])
            events["end_y"] = events["end_y"].fillna(events["y"])
        events["bodypart_name"] = StatsbombDataPreprocessor._infer_spadl_bodypart_name(events)
        events["success"] = events["success"].astype("boolean")
        emitted_mask = events["spadl_type"].notna()
        events.loc[emitted_mask & events["success"].isna(), "success"] = False
        events["result_name"] = StatsbombDataPreprocessor._infer_spadl_result_name(events)

        take_on = events["spadl_type"] == "take_on"
        events.loc[take_on, "receiver_id"] = pd.NA
        if "receiver_object_id" in events.columns:
            events.loc[take_on, "receiver_object_id"] = pd.NA

        selected = [
            "match_id",
            "original_event_id",
            "event_index",
            "period",
            "utc_timestamp",
            "match_clock",
            "match_clock_seconds",
            "team_id",
            "player_id",
            "object_id",
            "receiver_id",
            "receiver_object_id",
            "spadl_type",
            "x",
            "y",
            "end_x",
            "end_y",
            "bodypart_name",
            "result_name",
            "success",
            "home_score",
            "away_score",
            "score",
        ]
        if include_cdf_columns:
            selected.extend(["type", "sub_type", "outcome_detailed"])
        available = [column for column in selected if column in events.columns]
        output = events.loc[emitted_mask, available].copy().reset_index(drop=True)
        output = output.rename(
            columns={
                "period": "period_id",
                "match_clock_seconds": "time_seconds",
                "x": "start_x",
                "y": "start_y",
                "type": "cdf_type",
                "sub_type": "cdf_sub_type",
                "outcome_detailed": "cdf_outcome_detailed",
            }
        )
        output = output[output["player_id"].notna()].reset_index(drop=True)
        output["success"] = output["success"].fillna(False).astype(bool)
        for column in (
            "match_id",
            "original_event_id",
            "period_id",
            "team_id",
            "player_id",
            "object_id",
            "receiver_id",
            "receiver_object_id",
            "spadl_type",
            "bodypart_name",
            "result_name",
            "score",
        ):
            if column in output.columns:
                output[column] = output[column].astype("string")
        if include_cdf_columns:
            for column in ("cdf_type", "cdf_sub_type", "cdf_outcome_detailed"):
                output[column] = output[column].astype("string")
        ordered = [column for column in SPADL_ACTION_COLUMNS if column in output.columns]
        if include_cdf_columns:
            ordered.extend(
                column
                for column in ("cdf_type", "cdf_sub_type", "cdf_outcome_detailed")
                if column in output.columns
            )
        return output.loc[:, ordered]

    def preprocess_spadl_events(
        self,
        events_cdf: Optional[pd.DataFrame] = None,
        *,
        infer_end_coordinates: bool = True,
        include_cdf_columns: bool = True,
        include_extended_actions: bool = True,
        split_interception_passes: bool = True,
    ) -> pd.DataFrame:
        """Return SPADL-style actions, including explicit provider extensions.

        StatsBomb pass events whose subtype is ``Interception`` represent two
        on-ball actions. They are split by default, matching the compound-event
        handling used by the other provider adapters.
        """
        helper_columns = (
            "duel_type",
            "block_save_block",
            "goalkeeper_type",
            "card_type",
            "pass_height",
        )
        if events_cdf is None:
            events_cdf = self.preprocess_cdf_events(preserve_raw_columns=True)
        else:
            missing = [column for column in helper_columns if column not in events_cdf.columns]
            if missing:
                enriched = self.preprocess_cdf_events(preserve_raw_columns=True)
                events_cdf = events_cdf.merge(
                    enriched[["event_id", *missing]], on="event_id", how="left"
                )
        return self.to_spadl_events(
            events_cdf,
            infer_end_coordinates=infer_end_coordinates,
            include_cdf_columns=include_cdf_columns,
            include_extended_actions=include_extended_actions,
            split_interception_passes=split_interception_passes,
        )

    def preprocess_event_data(self) -> pd.DataFrame:
        """Backward-compatible event entry point used by existing providers."""
        return self.preprocess_spadl_events()

    # ------------------------------------------------------------------
    # StatsBomb 360 -> event-context tables (not tracking)
    # ------------------------------------------------------------------
    def _ensure_360_loaded(self) -> None:
        if self.raw_360 or self.three_sixty_path is None:
            return
        self.raw_360 = self.load_360_data(str(self.three_sixty_path))

    def preprocess_360_data(
        self,
        raw_360: Optional[list[dict]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw_360 is None:
            self._ensure_360_loaded()
            raw_360 = self.raw_360
        if not raw_360:
            return (
                pd.DataFrame(columns=THREE_SIXTY_FRAME_COLUMNS),
                pd.DataFrame(columns=THREE_SIXTY_OBJECT_COLUMNS),
            )

        event_lookup = self.events.set_index("event_id").to_dict("index")
        home_id = str(self.match_metadata.get("home_team_id"))
        away_id = str(self.match_metadata.get("away_team_id"))
        frame_rows = []
        object_rows = []
        for frame in raw_360:
            event_id = str(frame.get("event_uuid"))
            event = event_lookup.get(event_id, {})
            actor_team_id = str(event.get("team_id")) if pd.notna(event.get("team_id")) else pd.NA
            actor_player_id = (
                str(event.get("player_id")) if pd.notna(event.get("player_id")) else pd.NA
            )
            rotate = actor_team_id == away_id
            visible = frame.get("visible_area") or []
            visible_area: list[float] = []
            for index in range(0, len(visible) - 1, 2):
                x, y = self._transform_location(
                    [visible[index], visible[index + 1]], rotate=rotate
                )
                visible_area.extend([x, y])

            freeze_frame = frame.get("freeze_frame") or []
            has_actor = any(bool(obj.get("actor", False)) for obj in freeze_frame)
            frame_rows.append(
                {
                    "match_id": self.match_id,
                    "event_id": event_id,
                    "event_index": event.get("event_index", pd.NA),
                    "actor_team_id": actor_team_id,
                    "actor_player_id": actor_player_id,
                    "visible_area": visible_area,
                    "n_visible_players": len(freeze_frame),
                    "has_actor": has_actor,
                }
            )

            for object_index, observed in enumerate(freeze_frame):
                location = observed.get("location") or [None, None]
                x, y = self._transform_location(location, rotate=rotate)
                teammate = bool(observed.get("teammate", False))
                if actor_team_id == home_id:
                    team_id = home_id if teammate else away_id
                elif actor_team_id == away_id:
                    team_id = away_id if teammate else home_id
                else:
                    team_id = pd.NA
                is_actor = bool(observed.get("actor", False))
                object_rows.append(
                    {
                        "match_id": self.match_id,
                        "event_id": event_id,
                        "event_index": event.get("event_index", pd.NA),
                        "observed_object_index": object_index,
                        "team_id": team_id,
                        # The 360 feed deliberately omits identity for all
                        # non-actors. Never infer it from spatial proximity.
                        "player_id": actor_player_id if is_actor else pd.NA,
                        "teammate": teammate,
                        "actor": is_actor,
                        "keeper": bool(observed.get("keeper", False)),
                        "x": x,
                        "y": y,
                        "in_pitch": bool(
                            pd.notna(x)
                            and pd.notna(y)
                            and abs(float(x)) <= PITCH_X / 2.0
                            and abs(float(y)) <= PITCH_Y / 2.0
                        ),
                        "source_x": location[0] if len(location) > 0 else np.nan,
                        "source_y": location[1] if len(location) > 1 else np.nan,
                    }
                )

        frames = pd.DataFrame(frame_rows)
        objects = pd.DataFrame(object_rows)
        for frame in (frames, objects):
            for column in ("match_id", "event_id", "team_id", "player_id"):
                if column in frame.columns:
                    frame[column] = frame[column].astype("string")
            if "event_index" in frame.columns:
                frame["event_index"] = pd.to_numeric(
                    frame["event_index"], errors="coerce"
                ).astype("Int64")
        return (
            frames.loc[
                :,
                [
                    column
                    for column in THREE_SIXTY_FRAME_COLUMNS
                    if column in frames.columns
                ],
            ],
            objects.loc[
                :,
                [
                    column
                    for column in THREE_SIXTY_OBJECT_COLUMNS
                    if column in objects.columns
                ],
            ],
        )

    @staticmethod
    def load_tracking_data(*args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "StatsBomb Open Data has event-linked 360 snapshots, not continuous tracking. "
            "Use preprocess_360_data() instead."
        )

    def preprocess_tracking_data(self, *args, **kwargs):
        raise NotImplementedError(
            "StatsBomb Open Data has no continuous tracking stream. "
            "Use preprocess_360_data() for event-linked spatial context."
        )
