"""Hudl continuous tracking JSONL -> CDF-aligned tracking tables.

Hudl's tracking export is a single JSONL file per match.  It supplies player
coordinates by home/away side and shirt number rather than a provider player
identifier.  This adapter therefore preserves a stable side-and-jersey
``object_id`` and only assigns an event ``player_id`` when the event lineup
provides an exact shirt-number match.

The source does not declare pitch dimensions, a pitch orientation, possession,
or an in-play flag.  Coordinates retain their source scale by default; only a
per-period 180-degree rotation is applied so the home team is on the left.
Callers can supply separately provided physical pitch dimensions to normalize
coordinates to the shared 105 x 68 metre CDF reference. Unknown game-state
fields remain missing in the CDF outputs.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CDF_PERIOD_MAP
from .hudl_statsbomb_preprocessing import HudlStatsbombDataPreprocessor


CANONICAL_PITCH_LENGTH_M = 105.0
CANONICAL_PITCH_WIDTH_M = 68.0


def load_hudl_pitch_dimensions(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load Hudl companion pitch dimensions keyed by tracking match ID.

    The provider companion file uses ``wyscout_match_id`` for the tracking
    match ID and calls pitch width ``pitch_height_m``. The data file itself is
    intentionally external to this package; this helper only validates and
    parses its documented schema.
    """
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Pitch-dimensions file must contain a JSON list: {path}")

    dimensions: dict[str, tuple[float, float]] = {}
    for row_number, row in enumerate(payload, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Pitch-dimensions row {row_number} is not an object")
        try:
            match_id = str(row["wyscout_match_id"])
            length_m = float(row["pitch_length_m"])
            width_m = float(row["pitch_height_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Pitch-dimensions row {row_number} must provide "
                "wyscout_match_id, pitch_length_m, and pitch_height_m"
            ) from error
        if not np.isfinite(length_m) or not np.isfinite(width_m) or length_m <= 0 or width_m <= 0:
            raise ValueError(f"Invalid pitch dimensions for tracking match {match_id}: {length_m} x {width_m}")
        if match_id in dimensions:
            raise ValueError(f"Duplicate tracking match ID in pitch-dimensions file: {match_id}")
        dimensions[match_id] = (length_m, width_m)
    return dimensions


def normalize_hudl_tracking_coordinates(
    tracking: pd.DataFrame,
    *,
    source_pitch_length_m: float,
    source_pitch_width_m: float,
    target_pitch_length_m: float = CANONICAL_PITCH_LENGTH_M,
    target_pitch_width_m: float = CANONICAL_PITCH_WIDTH_M,
) -> pd.DataFrame:
    """Scale centre-origin Hudl tracking coordinates to a target pitch size.

    This function rescales only the active CDF ``x``/``y`` coordinates. Raw
    provider fields such as ``source_x`` and ``source_y`` are preserved, so
    callers can audit the transformation. Orientation is handled separately
    by :class:`HudlTrackingDataPreprocessor` before this function is called.
    """
    source_pitch_length_m = float(source_pitch_length_m)
    source_pitch_width_m = float(source_pitch_width_m)
    target_pitch_length_m = float(target_pitch_length_m)
    target_pitch_width_m = float(target_pitch_width_m)
    values = (
        source_pitch_length_m,
        source_pitch_width_m,
        target_pitch_length_m,
        target_pitch_width_m,
    )
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("Source and target pitch dimensions must be finite positive values")
    if not {"x", "y"}.issubset(tracking.columns):
        raise ValueError("Tracking table must contain x and y columns")

    output = tracking.copy()
    output["x"] = pd.to_numeric(output["x"], errors="coerce") * target_pitch_length_m / source_pitch_length_m
    output["y"] = pd.to_numeric(output["y"], errors="coerce") * target_pitch_width_m / source_pitch_width_m
    return output


class HudlTrackingDataPreprocessor(HudlStatsbombDataPreprocessor):
    """Load paired Hudl flattened events and continuous tracking for one match.

    ``input_tracking`` is a geometrically complete candidate set: frames with
    exactly eleven observed players for each side.  It is deliberately *not*
    an in-play table because Hudl does not provide an in-play/ball-state signal.
    Its ``ball_status`` and ``ball_poss_team_id`` values remain missing.
    """

    SOURCE_FPS = 25.0
    _REQUIRED_HEADER_KEYS = {"gameRefId", "generatedTime", "metadata", "version"}
    _REQUIRED_FRAME_KEYS = {
        "awayPlayers",
        "balls",
        "frameNum",
        "homePlayers",
        "period",
        "periodElapsedTime",
        "periodGameClockTime",
        "referees",
        "videoTimeMs",
    }

    def __init__(
        self,
        event_path: str,
        tracking_path: str,
        *,
        required_players_per_team: int = 11,
    ):
        super().__init__(event_path)
        if required_players_per_team != 11:
            raise ValueError("Hudl input tracking currently requires exactly 11 players per team")

        self.tracking_path = str(Path(tracking_path))
        self.required_players_per_team = required_players_per_team
        self.raw_tracking_header = self.load_tracking_header(self.tracking_path)
        self._validate_tracking_match(self.raw_tracking_header)
        self.tracking_match_id = str(self.raw_tracking_header["metadata"]["matchId"])
        self.match_metadata.update(
            {
                "tracking_match_id": self.tracking_match_id,
                "tracking_source_pitch_length": pd.NA,
                "tracking_source_pitch_width": pd.NA,
                "tracking_coordinate_scale_verified": False,
                "tracking_coordinate_orientation": "home_left_rotated_source_scale",
            }
        )
        self.fps = self.SOURCE_FPS
        self.orientation_rotation_by_period: dict[int, bool] = {}
        self._tracking_quality = pd.DataFrame()
        self.tracking = self.load_tracking_data()
        self.tracking_long = pd.DataFrame()

    # ------------------------------------------------------------------
    # Header and match-pair validation
    # ------------------------------------------------------------------
    @staticmethod
    def _team_key(value: object) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        aliases = {
            "internazionale": "inter",
            "inter milan": "inter",
            "ac milan": "milan",
            "as roma": "roma",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def load_tracking_header(cls, tracking_path: str | Path) -> dict[str, Any]:
        path = Path(tracking_path)
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        if not first_line.strip():
            raise ValueError(f"Hudl tracking file has no header: {path}")
        header = json.loads(first_line)
        if not isinstance(header, Mapping) or not cls._REQUIRED_HEADER_KEYS.issubset(header):
            raise ValueError(f"Invalid Hudl tracking header: {path}")
        metadata = header.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Hudl tracking header metadata is missing: {path}")
        required_metadata = {"matchId", "scheduledTime", "label"}
        if not required_metadata.issubset(metadata):
            raise ValueError(f"Hudl tracking header is missing match identity fields: {path}")
        return dict(header)

    @classmethod
    def _tracking_label_identity(cls, label: object) -> tuple[str, str, int, int]:
        try:
            teams, score = str(label).rsplit(",", 1)
            home, away = (value.strip() for value in teams.split(" - ", 1))
            home_score, away_score = (int(value.strip()) for value in score.split(" - ", 1))
        except (ValueError, AttributeError) as error:
            raise ValueError(f"Cannot parse Hudl tracking match label: {label!r}") from error
        return home, away, home_score, away_score

    def _validate_tracking_match(self, header: Mapping[str, Any]) -> None:
        metadata = header["metadata"]
        home, away, home_score, away_score = self._tracking_label_identity(metadata["label"])
        event_date = pd.Timestamp(self.match_metadata["kickoff_time"]).date().isoformat()
        tracking_date = str(metadata["scheduledTime"])[:10]
        event_home = self.match_metadata["home_team_name"]
        event_away = self.match_metadata["away_team_name"]
        event_score = (
            int(self.match_metadata["final_home_score"]),
            int(self.match_metadata["final_away_score"]),
        )
        identity_matches = (
            event_date == tracking_date
            and self._team_key(event_home) == self._team_key(home)
            and self._team_key(event_away) == self._team_key(away)
            and event_score == (home_score, away_score)
        )
        if not identity_matches:
            raise ValueError(
                "Event and tracking paths do not identify the same match: "
                f"event={event_date} {event_home} vs {event_away} {event_score[0]}-{event_score[1]}, "
                f"tracking={tracking_date} {home} vs {away} {home_score}-{away_score}"
            )

    # ------------------------------------------------------------------
    # Streaming raw tracking loader
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_coordinate(observed: Mapping[str, Any]) -> bool:
        return observed.get("x") is not None and observed.get("y") is not None

    def _jersey_to_player_id(self) -> dict[tuple[str, str], str]:
        mapping: dict[tuple[str, str], str] = {}
        for row in self.lineup.itertuples(index=False):
            if pd.isna(row.uniform_number) or pd.isna(row.player_id):
                continue
            side = str(row.home_away)
            if side not in {"home", "away"}:
                continue
            jersey = str(int(row.uniform_number))
            key = (side, jersey)
            player_id = str(row.player_id)
            if key in mapping and mapping[key] != player_id:
                raise ValueError(f"Ambiguous event lineup shirt number: {key}")
            mapping[key] = player_id
        return mapping

    def _goalkeeper_jerseys(self) -> dict[str, str]:
        goalkeepers: dict[str, str] = {}
        gk_rows = self.lineup.loc[
            self.lineup["playing_position"].eq("GK")
            & self.lineup["uniform_number"].notna(),
            ["home_away", "uniform_number"],
        ]
        for row in gk_rows.itertuples(index=False):
            side = str(row.home_away)
            if side in goalkeepers:
                raise ValueError(f"Multiple starting goalkeepers in event lineup for {side}")
            goalkeepers[side] = str(int(row.uniform_number))
        if set(goalkeepers) != {"home", "away"}:
            raise ValueError("Hudl tracking orientation requires home and away starting goalkeeper jerseys")
        return goalkeepers

    def _detect_home_left_rotation(self, tracking: pd.DataFrame) -> dict[int, bool]:
        goalkeepers = self._goalkeeper_jerseys()
        rotations: dict[int, bool] = {}
        for period_id, period in tracking.groupby("period_id", sort=True):
            home_x = pd.to_numeric(period.get(f"home_{goalkeepers['home']}_x"), errors="coerce")
            away_x = pd.to_numeric(period.get(f"away_{goalkeepers['away']}_x"), errors="coerce")
            home_median = home_x.median()
            away_median = away_x.median()
            if pd.isna(home_median) or pd.isna(away_median):
                raise ValueError(f"Cannot establish Hudl tracking orientation in period {period_id}")
            if abs(float(home_median) - float(away_median)) < 1.0:
                raise ValueError(f"Ambiguous Hudl goalkeeper orientation in period {period_id}")
            rotations[int(period_id)] = bool(home_median > away_median)
        return rotations

    @staticmethod
    def _rotate_wide_coordinates(tracking: pd.DataFrame, rotations: Mapping[int, bool]) -> pd.DataFrame:
        output = tracking.copy()
        coordinate_columns = [
            column
            for column in output.columns
            if column == "ball_x"
            or column == "ball_y"
            or (column.startswith(("home_", "away_")) and column.endswith(("_x", "_y")))
        ]
        for period_id, rotate in rotations.items():
            if rotate:
                mask = output["period_id"].eq(period_id)
                output.loc[mask, coordinate_columns] = -output.loc[mask, coordinate_columns]
        return output

    def load_tracking_data(self) -> pd.DataFrame:
        jersey_to_player = self._jersey_to_player_id()
        rows: list[dict[str, Any]] = []
        quality_rows: list[dict[str, Any]] = []
        period_start_video_ms: dict[int, float] = {}

        with Path(self.tracking_path).open("r", encoding="utf-8") as handle:
            next(handle)  # Header was parsed and validated in __init__.
            for line_number, line in enumerate(handle, start=2):
                if not line.strip():
                    continue
                frame = json.loads(line)
                if not isinstance(frame, Mapping) or not self._REQUIRED_FRAME_KEYS.issubset(frame):
                    raise ValueError(f"Invalid Hudl tracking frame at line {line_number}")
                period_id = int(frame["period"])
                video_time_ms = float(frame["videoTimeMs"])
                period_start_video_ms.setdefault(period_id, video_time_ms)
                row: dict[str, Any] = {
                    "frame_id": int(frame["frameNum"]),
                    "period_id": period_id,
                    # Continuous and strictly monotonic at the verified 25 Hz;
                    # provider game-clock fields are retained separately below.
                    "timestamp": (video_time_ms - period_start_video_ms[period_id]) / 1000.0,
                    "utc_timestamp": pd.NaT,
                    # The provider exposes neither an in-play state nor possession.
                    # A string sentinel is used internally so the base finalizer can
                    # run; it becomes missing in the returned CDF column.
                    "ball_state": "unknown",
                    "ball_owning_team_id": pd.NA,
                    "source_video_time_ms": video_time_ms,
                    "source_period_elapsed_time": frame["periodElapsedTime"],
                    "source_period_game_clock_time": frame["periodGameClockTime"],
                    "home_player_count": 0,
                    "away_player_count": 0,
                }
                for side, source_key in (("home", "homePlayers"), ("away", "awayPlayers")):
                    seen_jerseys: set[str] = set()
                    for observed in frame[source_key]:
                        if not isinstance(observed, Mapping) or not self._valid_coordinate(observed):
                            continue
                        jersey = observed.get("jerseyNum")
                        if jersey is None:
                            continue
                        jersey_key = str(jersey)
                        if jersey_key in seen_jerseys:
                            raise ValueError(
                                f"Duplicate {side} jersey {jersey_key!r} in frame {row['frame_id']}"
                            )
                        seen_jerseys.add(jersey_key)
                        object_id = f"{side}_{jersey_key}"
                        row[f"{object_id}_x"] = float(observed["x"])
                        row[f"{object_id}_y"] = float(observed["y"])
                        row[f"{object_id}_source_visibility"] = observed.get("visibility", pd.NA)
                        row[f"{object_id}_source_confidence"] = observed.get("confidence", pd.NA)
                        row[f"{side}_player_count"] += 1
                        quality_rows.append(
                            {
                                "period_id": period_id,
                                "frame_id": row["frame_id"],
                                "object_id": object_id,
                                "source_x": float(observed["x"]),
                                "source_y": float(observed["y"]),
                                "source_z": np.nan,
                                "source_speed": observed.get("speed", pd.NA),
                                "source_vx": observed.get("vx", pd.NA),
                                "source_vy": observed.get("vy", pd.NA),
                                "source_visibility": observed.get("visibility", pd.NA),
                                "source_confidence": observed.get("confidence", pd.NA),
                                "player_id": jersey_to_player.get((side, jersey_key), pd.NA),
                            }
                        )
                balls = [
                    observed
                    for observed in frame["balls"]
                    if isinstance(observed, Mapping) and self._valid_coordinate(observed)
                ]
                if len(balls) > 1:
                    raise ValueError(f"Multiple observed balls in frame {row['frame_id']}")
                if balls:
                    ball = balls[0]
                    row["ball_x"] = float(ball["x"])
                    row["ball_y"] = float(ball["y"])
                    row["ball_z"] = pd.to_numeric(ball.get("z"), errors="coerce")
                    quality_rows.append(
                        {
                            "period_id": period_id,
                            "frame_id": row["frame_id"],
                            "object_id": "ball",
                            "source_x": float(ball["x"]),
                            "source_y": float(ball["y"]),
                            "source_z": pd.to_numeric(ball.get("z"), errors="coerce"),
                            "source_speed": ball.get("speed", pd.NA),
                            "source_vx": ball.get("vx", pd.NA),
                            "source_vy": ball.get("vy", pd.NA),
                            "source_visibility": ball.get("visibility", pd.NA),
                            "source_confidence": ball.get("confidence", pd.NA),
                            "player_id": pd.NA,
                        }
                    )
                rows.append(row)

        tracking = pd.DataFrame(rows)
        if tracking.empty:
            raise ValueError(f"Hudl tracking file has no frames: {self.tracking_path}")
        self.orientation_rotation_by_period = self._detect_home_left_rotation(tracking)
        tracking = self._rotate_wide_coordinates(tracking, self.orientation_rotation_by_period)
        self._tracking_quality = pd.DataFrame(quality_rows)
        return tracking.sort_values(["period_id", "frame_id"], kind="mergesort").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Wide source frames -> CDF-compatible long tables
    # ------------------------------------------------------------------
    def _attach_source_fields(self, tracking_cdf: pd.DataFrame) -> pd.DataFrame:
        if tracking_cdf.empty:
            return tracking_cdf
        frame_metadata = self.tracking.loc[
            :,
            [
                "period_id",
                "frame_id",
                "source_video_time_ms",
                "source_period_elapsed_time",
                "source_period_game_clock_time",
                "home_player_count",
                "away_player_count",
            ],
        ].copy()
        frame_metadata["period"] = frame_metadata["period_id"].map(CDF_PERIOD_MAP)
        output = tracking_cdf.merge(
            frame_metadata.drop(columns="period_id"),
            on=["period", "frame_id"],
            how="left",
            validate="many_to_one",
        )
        if self._tracking_quality.empty:
            return output
        quality = self._tracking_quality.copy()
        quality["period"] = quality["period_id"].map(CDF_PERIOD_MAP)
        quality["object_id"] = quality["object_id"].astype("string")
        source_columns = [
            "period",
            "frame_id",
            "object_id",
            "source_x",
            "source_y",
            "source_z",
            "source_speed",
            "source_vx",
            "source_vy",
            "source_visibility",
            "source_confidence",
        ]
        return output.merge(
            quality.loc[:, source_columns],
            on=["period", "frame_id", "object_id"],
            how="left",
            validate="one_to_one",
        )

    def preprocess_tracking_data(
        self,
        apply_kinematic_correction: bool = False,
        *,
        source_pitch_length_m: float | None = None,
        source_pitch_width_m: float | None = None,
        target_pitch_length_m: float = CANONICAL_PITCH_LENGTH_M,
        target_pitch_width_m: float = CANONICAL_PITCH_WIDTH_M,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return all observed tracking and complete-team candidate frames.

        ``ball_status`` and ``ball_poss_team_id`` remain missing in both
        outputs.  The second output is filtered only by the agreed 11-v-11
        geometric completeness criterion, not by an unavailable in-play flag.

        By default, coordinates retain the provider source scale. To normalize
        to a target pitch (105 x 68 m by default), provide both source physical
        pitch dimensions. The source and target coordinates are centre-origin,
        and raw provider ``source_x``/``source_y`` fields remain available for
        audit.
        """
        if (source_pitch_length_m is None) != (source_pitch_width_m is None):
            raise ValueError("source_pitch_length_m and source_pitch_width_m must be provided together")
        raw_tracking, _ = self._finalize_tracking_output(
            self.tracking,
            fps=self.fps,
            apply_kinematic_correction=apply_kinematic_correction,
        )
        raw_tracking = self._attach_source_fields(raw_tracking)
        if source_pitch_length_m is not None and source_pitch_width_m is not None:
            raw_tracking = normalize_hudl_tracking_coordinates(
                raw_tracking,
                source_pitch_length_m=source_pitch_length_m,
                source_pitch_width_m=source_pitch_width_m,
                target_pitch_length_m=target_pitch_length_m,
                target_pitch_width_m=target_pitch_width_m,
            )
            self.match_metadata.update(
                {
                    "tracking_source_pitch_length": float(source_pitch_length_m),
                    "tracking_source_pitch_width": float(source_pitch_width_m),
                    "tracking_coordinate_scale_verified": True,
                    "tracking_coordinate_orientation": "home_left_rotated_normalized",
                    "tracking_target_pitch_length": float(target_pitch_length_m),
                    "tracking_target_pitch_width": float(target_pitch_width_m),
                }
            )
        # Base finalization maps the internal "unknown" sentinel to NaN. Keep
        # the public CDF game-state fields explicitly missing.
        raw_tracking["ball_status"] = pd.Series(pd.NA, index=raw_tracking.index, dtype="boolean")
        raw_tracking["ball_poss_team_id"] = pd.Series(pd.NA, index=raw_tracking.index, dtype="string")

        complete_frames = self.tracking.loc[
            self.tracking["home_player_count"].eq(self.required_players_per_team)
            & self.tracking["away_player_count"].eq(self.required_players_per_team),
            ["period_id", "frame_id"],
        ].copy()
        complete_frames["period"] = complete_frames["period_id"].map(CDF_PERIOD_MAP)
        input_tracking = raw_tracking.merge(
            complete_frames.loc[:, ["period", "frame_id"]].drop_duplicates(),
            on=["period", "frame_id"],
            how="inner",
            validate="many_to_one",
        )
        self.tracking_long = raw_tracking
        return raw_tracking.reset_index(drop=True), input_tracking.reset_index(drop=True)
