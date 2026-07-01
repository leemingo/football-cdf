from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from .base import BaseEventTrackingPreprocessor
from .constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y
from .skillcorner_paths import META_FILENAMES, find_match_dir


SKILLCORNER_POSITION_MAPPING = {
    None: None,
    "AM": "CAM",
    "CB": "CB",
    "CF": "CF",
    "CM": "CM",
    "DM": "CDM",
    "GK": "GK",
    "LB": "LB",
    "LCB": "LCB",
    "LDM": "LDM",
    "LF": "LCF",
    "LM": "LM",
    "LW": "LWF",
    "LWB": "LWB",
    "RB": "RB",
    "RCB": "RCB",
    "RDM": "RDM",
    "RF": "RCF",
    "RM": "RM",
    "RW": "RWF",
    "RWB": "RWB",
    "SUB": "SUB",
}


class SkillcornerDataPreprocessor(BaseEventTrackingPreprocessor):
    """SkillCorner metadata/lineup loader.

    The provider-specific event and tracking preprocessing steps are not
    implemented yet. This class currently focuses on building CDF-aligned
    match metadata from ``match_meta.json`` / ``match.json`` bundles.
    """

    META_FILENAMES = META_FILENAMES
    EVENT_FILENAME = "dynamic_events.csv"
    TRACKING_FILENAME = "tracking.jsonl"
    UNDETECTED_OUTLIER_TOL = 3.0

    def __init__(
        self,
        root_dir: str,
        match_id: str,
        load_tracking: bool = False,
    ):
        super().__init__()
        self.match_id = str(match_id)
        self.match_path = self._locate_match_path(root_dir, self.match_id)

        self.meta_path = self._resolve_existing_file(self.match_path, self.META_FILENAMES)
        self.event_path = self.match_path / self.EVENT_FILENAME
        self.tracking_path = self.match_path / self.TRACKING_FILENAME

        self.raw_metadata = self.load_raw_metadata(str(self.meta_path))
        self.match_metadata = self.extract_match_metadata(self.raw_metadata)
        self.lineup = self.load_lineup_data(self.raw_metadata, self.match_metadata)
        self.events = pd.DataFrame()
        self.tracking = pd.DataFrame()
        self.tracking_long = pd.DataFrame()
        self.fps = self._estimate_fps(self.raw_metadata)

        if load_tracking:
            self.tracking_long = self.load_tracking_long_data(
                self.tracking_path,
                self.raw_metadata,
                self.lineup,
                self.match_metadata,
                self.fps,
            )
            self.tracking = self.tracking_long_to_wide(self.tracking_long, self.lineup)

    @classmethod
    def _locate_match_path(
        cls,
        root_dir: str,
        match_id: str,
    ) -> Path:
        existing = find_match_dir(root_dir, match_id)
        if existing is not None:
            return existing

        raise FileNotFoundError(
            f"Could not find a local SkillCorner match directory for match_id={match_id!r} "
            f"under {root_dir}. Download SkillCorner Open Data or point root_dir at a "
            "local match-bundle directory."
        )

    @staticmethod
    def _resolve_existing_file(match_path: Path, candidates: Iterable[str]) -> Path:
        for filename in candidates:
            candidate = match_path / filename
            if candidate.exists():
                return candidate
        joined = ", ".join(str(match_path / name) for name in candidates)
        raise FileNotFoundError(f"Required SkillCorner file is missing. Tried: {joined}")

    @staticmethod
    def _coalesce_id(id_value, name_value, field_name: str):
        if id_value is None and name_value is not None:
            warnings.warn(
                f"{field_name} missing in raw metadata; using {field_name.replace('_id', '_name')} instead",
                stacklevel=2,
            )
            return name_value
        return id_value

    @staticmethod
    def _display_team_name(team: dict) -> object:
        if not team:
            return pd.NA
        return team.get("short_name") or team.get("name") or pd.NA

    @staticmethod
    def _player_name(player: dict) -> object:
        short_name = player.get("short_name")
        if short_name:
            return short_name
        first_name = player.get("first_name") or ""
        last_name = player.get("last_name") or ""
        full_name = f"{first_name} {last_name}".strip()
        return full_name or pd.NA

    @staticmethod
    def _normalize_position(player_role: dict | None) -> object:
        role = player_role or {}
        acronym = role.get("acronym")
        return SKILLCORNER_POSITION_MAPPING.get(acronym, acronym or role.get("name") or pd.NA)

    @staticmethod
    def _build_object_id(home_away: object, uniform_number: object) -> object:
        if pd.isna(home_away) or pd.isna(uniform_number):
            return pd.NA
        try:
            number = int(uniform_number)
        except (TypeError, ValueError):
            return pd.NA
        return f"{home_away}_{number}"

    @staticmethod
    def _extract_period_details(raw_metadata: dict) -> dict:
        period_rows = {}
        periods = raw_metadata.get("match_periods") or []
        home_team_side = raw_metadata.get("home_team_side") or []

        for idx, period in enumerate(periods):
            period_id = period.get("period")
            if period_id is None:
                continue

            side_value = home_team_side[idx] if idx < len(home_team_side) else pd.NA
            period_rows[f"period_{period_id}_name"] = period.get("name", pd.NA)
            period_rows[f"period_{period_id}_start_frame"] = period.get("start_frame", pd.NA)
            period_rows[f"period_{period_id}_end_frame"] = period.get("end_frame", pd.NA)
            period_rows[f"period_{period_id}_duration_frames"] = period.get("duration_frames", pd.NA)
            period_rows[f"period_{period_id}_duration_minutes"] = period.get("duration_minutes", pd.NA)
            period_rows[f"home_team_side_period_{period_id}"] = side_value

        return period_rows

    @staticmethod
    def _to_cdf_play_direction(side_value: object) -> object:
        mapping = {
            "left_to_right": "leftright",
            "right_to_left": "rightleft",
        }
        return mapping.get(side_value, pd.NA)

    @classmethod
    def _build_play_direction(cls, raw_metadata: dict) -> dict:
        play_direction = {}
        periods = raw_metadata.get("match_periods") or []
        home_team_side = raw_metadata.get("home_team_side") or []

        for idx, period in enumerate(periods):
            period_id = period.get("period")
            period_name = CDF_PERIOD_MAP.get(period_id)
            if period_name is None:
                continue

            side_value = home_team_side[idx] if idx < len(home_team_side) else pd.NA
            cdf_direction = cls._to_cdf_play_direction(side_value)
            if pd.isna(cdf_direction):
                continue

            play_direction[period_name] = cdf_direction

        return play_direction

    @staticmethod
    def _estimate_fps(raw_metadata: dict) -> float:
        fps_values = []
        for period in raw_metadata.get("match_periods") or []:
            duration_frames = period.get("duration_frames")
            duration_minutes = period.get("duration_minutes")
            if duration_frames in (None, 0) or duration_minutes in (None, 0):
                continue
            fps_values.append(float(duration_frames) / (float(duration_minutes) * 60.0))

        if not fps_values:
            return float("nan")
        return round(float(pd.Series(fps_values).median()), 3)

    @staticmethod
    def _infer_starting(player: dict, *, first_period_start_frame: object) -> bool:
        start_time = player.get("start_time")
        if start_time == "00:00:00":
            return True

        total_playing_time = (player.get("playing_time") or {}).get("total") or {}
        start_frame = total_playing_time.get("start_frame")
        if start_frame is None or pd.isna(first_period_start_frame):
            return False

        return int(start_frame) <= int(first_period_start_frame)

    @staticmethod
    def load_raw_metadata(meta_path: str) -> dict:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def extract_match_metadata(raw_metadata: dict) -> dict:
        competition_edition = raw_metadata.get("competition_edition") or {}
        competition = competition_edition.get("competition") or {}
        season = competition_edition.get("season") or {}
        competition_round = raw_metadata.get("competition_round") or {}
        home_team = raw_metadata.get("home_team") or {}
        away_team = raw_metadata.get("away_team") or {}
        stadium = raw_metadata.get("stadium") or {}

        final_home_score = raw_metadata.get("home_team_score", pd.NA)
        final_away_score = raw_metadata.get("away_team_score", pd.NA)
        final_score = pd.NA
        if pd.notna(final_home_score) and pd.notna(final_away_score):
            final_score = f"{final_home_score}:{final_away_score}"

        metadata = {
            "competition_id": SkillcornerDataPreprocessor._coalesce_id(
                competition.get("id"),
                competition.get("name"),
                "competition_id",
            ),
            "competition_name": competition.get("name"),
            "season_id": SkillcornerDataPreprocessor._coalesce_id(
                season.get("id"),
                season.get("name"),
                "season_id",
            ),
            "season_name": season.get("name"),
            "match_id": (
                str(raw_metadata.get("id"))
                if raw_metadata.get("id") is not None
                else pd.NA
            ),
            "kickoff_time": raw_metadata.get("date_time"),
            "play_direction": SkillcornerDataPreprocessor._build_play_direction(raw_metadata),
            "home_team_id": (
                str(home_team.get("id"))
                if home_team.get("id") is not None
                else pd.NA
            ),
            "home_team_name": SkillcornerDataPreprocessor._display_team_name(home_team),
            "away_team_id": (
                str(away_team.get("id"))
                if away_team.get("id") is not None
                else pd.NA
            ),
            "away_team_name": SkillcornerDataPreprocessor._display_team_name(away_team),
            "stadium_id": SkillcornerDataPreprocessor._coalesce_id(
                stadium.get("id"),
                stadium.get("name"),
                "stadium_id",
            ),
            "stadium_name": stadium.get("name", pd.NA),
            "pitch_length": raw_metadata.get("pitch_length"),
            "pitch_width": raw_metadata.get("pitch_width"),
            "final_home_score": final_home_score,
            "final_away_score": final_away_score,
            "final_score": final_score,
            "vendor_name": "SkillCorner",
            "vendor_version": pd.NA,
            "cdf_version": "v1",
            "competition_area": competition.get("area", pd.NA),
            "competition_gender": competition.get("gender", pd.NA),
            "competition_age_group": competition.get("age_group", pd.NA),
            "competition_round_id": competition_round.get("id", pd.NA),
            "competition_round_name": competition_round.get("name", pd.NA),
            "competition_round_number": competition_round.get("round_number", pd.NA),
            "match_status": raw_metadata.get("status", pd.NA),
            "home_team_official_name": home_team.get("name", pd.NA),
            "home_team_acronym": home_team.get("acronym", pd.NA),
            "away_team_official_name": away_team.get("name", pd.NA),
            "away_team_acronym": away_team.get("acronym", pd.NA),
            "stadium_city": stadium.get("city", pd.NA),
            "ball_object_id": (
                str((raw_metadata.get("ball") or {}).get("trackable_object"))
                if (raw_metadata.get("ball") or {}).get("trackable_object") is not None
                else pd.NA
            ),
            "source_fps": SkillcornerDataPreprocessor._estimate_fps(raw_metadata),
        }
        metadata.update(SkillcornerDataPreprocessor._extract_period_details(raw_metadata))
        return metadata

    @staticmethod
    def load_lineup_data(raw_metadata: dict, match_metadata: dict) -> pd.DataFrame:
        home_team = raw_metadata.get("home_team") or {}
        away_team = raw_metadata.get("away_team") or {}
        home_team_id = str(home_team.get("id")) if home_team.get("id") is not None else None
        away_team_id = str(away_team.get("id")) if away_team.get("id") is not None else None
        team_names = {
            home_team_id: SkillcornerDataPreprocessor._display_team_name(home_team),
            away_team_id: SkillcornerDataPreprocessor._display_team_name(away_team),
        }
        first_period_start_frame = match_metadata.get("period_1_start_frame", pd.NA)

        lineup_rows = []
        for player in raw_metadata.get("players") or []:
            role = player.get("player_role") or {}
            team_id = str(player.get("team_id")) if player.get("team_id") is not None else pd.NA
            if team_id == home_team_id:
                home_away = "home"
            elif team_id == away_team_id:
                home_away = "away"
            else:
                home_away = pd.NA

            trackable_object = player.get("trackable_object")
            lineup_rows.append(
                {
                    "team_id": team_id,
                    "team_name": team_names.get(team_id, pd.NA),
                    "home_away": home_away,
                    "player_id": (
                        str(player.get("id"))
                        if player.get("id") is not None
                        else pd.NA
                    ),
                    "uniform_number": player.get("number", pd.NA),
                    "object_id": SkillcornerDataPreprocessor._build_object_id(
                        home_away,
                        player.get("number", pd.NA),
                    ),
                    "player_name": SkillcornerDataPreprocessor._player_name(player),
                    "playing_position": SkillcornerDataPreprocessor._normalize_position(role),
                    "starting": SkillcornerDataPreprocessor._infer_starting(
                        player,
                        first_period_start_frame=first_period_start_frame,
                    ),
                    "trackable_object_id": (
                        str(trackable_object)
                        if trackable_object is not None
                        else pd.NA
                    ),
                    "team_player_id": (
                        str(player.get("team_player_id"))
                        if player.get("team_player_id") is not None
                        else pd.NA
                    ),
                    "player_role_id": role.get("id", pd.NA),
                    "player_role_name": role.get("name", pd.NA),
                    "player_role_acronym": role.get("acronym", pd.NA),
                    "player_position_group": role.get("position_group", pd.NA),
                    "start_time": player.get("start_time", pd.NA),
                    "end_time": player.get("end_time", pd.NA),
                    "yellow_card": player.get("yellow_card", pd.NA),
                    "red_card": player.get("red_card", pd.NA),
                    "goal": player.get("goal", pd.NA),
                    "own_goal": player.get("own_goal", pd.NA),
                    "injured": player.get("injured", pd.NA),
                    "gender": player.get("gender", pd.NA),
                    "birthday": player.get("birthday", pd.NA),
                }
            )

        lineup_df = pd.DataFrame(lineup_rows)
        if not lineup_df.empty:
            lineup_df["home_away"] = pd.Categorical(
                lineup_df["home_away"],
                categories=["home", "away"],
                ordered=True,
            )
            lineup_df = lineup_df.sort_values(
                by=["home_away", "starting", "uniform_number", "player_name"],
                ascending=[True, False, True, True],
                ignore_index=True,
            )

        if not lineup_df.empty:
            for column in (
                "team_id",
                "player_id",
                "object_id",
                "trackable_object_id",
                "team_player_id",
            ):
                if column in lineup_df.columns:
                    lineup_df[column] = lineup_df[column].astype("string")

            if "uniform_number" in lineup_df.columns:
                lineup_df["uniform_number"] = pd.to_numeric(
                    lineup_df["uniform_number"], errors="coerce"
                ).astype("Int64")
            if "starting" in lineup_df.columns:
                lineup_df["starting"] = lineup_df["starting"].fillna(False).astype(bool)

        return lineup_df

    @staticmethod
    def _normalize_kickoff_time(kickoff_time: object) -> pd.Timestamp:
        timestamp = pd.to_datetime(kickoff_time, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return pd.NaT
        return timestamp.tz_convert("UTC").tz_localize(None)

    @staticmethod
    def _build_period_frame_map(raw_metadata: dict) -> dict[int, dict[str, object]]:
        period_frame_map: dict[int, dict[str, object]] = {}
        for period in raw_metadata.get("match_periods") or []:
            period_id = period.get("period")
            if period_id is None:
                continue
            period_frame_map[int(period_id)] = {
                "start_frame": period.get("start_frame"),
                "end_frame": period.get("end_frame"),
                "duration_frames": period.get("duration_frames"),
            }
        return period_frame_map

    @staticmethod
    def _rescale_tracking_coordinate(value: object, *, raw_pitch_size: float, target_pitch_size: float) -> object:
        if value is None or pd.isna(value) or raw_pitch_size in (None, 0) or pd.isna(raw_pitch_size):
            return pd.NA
        return float(value) * (float(target_pitch_size) / float(raw_pitch_size))

    @staticmethod
    def _frame_seconds(frame_id: int, start_frame: int, fps: float) -> float:
        return max(0.0, (int(frame_id) - int(start_frame)) / float(fps))

    @staticmethod
    def _map_possession_group_to_team_id(
        group: object,
        *,
        home_team_id: object,
        away_team_id: object,
    ) -> object:
        if group == "home team":
            return home_team_id
        if group == "away team":
            return away_team_id
        return pd.NA

    @staticmethod
    def _infer_ball_state(possession: dict, ball_data: dict) -> str:
        if possession.get("group") in {"home team", "away team"}:
            return "alive"
        if possession.get("player_id") is not None:
            return "alive"
        if ball_data.get("is_detected") is True:
            return "alive"
        return "dead"

    @staticmethod
    def _order_tracking_columns(tracking_df: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        fixed_cols = ["period_id", "timestamp", "frame_id", "utc_timestamp", "ball_state", "ball_owning_team_id"]
        metric_order = ["x", "y", "z"]

        def object_metric_columns(object_id: str) -> list[str]:
            return [
                f"{object_id}_{metric}"
                for metric in metric_order
                if f"{object_id}_{metric}" in tracking_df.columns
            ]

        lineup_order = lineup.sort_values(["home_away", "uniform_number"], kind="mergesort")
        home_object_ids = lineup_order.loc[lineup_order["home_away"] == "home", "object_id"].dropna().tolist()
        away_object_ids = lineup_order.loc[lineup_order["home_away"] == "away", "object_id"].dropna().tolist()

        ordered_dynamic_cols = []
        ordered_dynamic_cols.extend(object_metric_columns("ball"))
        for object_id in home_object_ids:
            ordered_dynamic_cols.extend(object_metric_columns(object_id))
        for object_id in away_object_ids:
            ordered_dynamic_cols.extend(object_metric_columns(object_id))

        remaining_cols = [
            col for col in tracking_df.columns
            if col not in fixed_cols and col not in ordered_dynamic_cols
        ]

        return tracking_df.loc[:, fixed_cols + ordered_dynamic_cols + remaining_cols]

    @staticmethod
    def _mask_undetected_outliers(
        tracking_long: pd.DataFrame,
        *,
        x_limit: float,
        y_limit: float,
        tolerance: float,
    ) -> pd.DataFrame:
        if tracking_long.empty or "is_detected" not in tracking_long.columns:
            return tracking_long

        tracking_long = tracking_long.copy()
        undetected_mask = tracking_long["is_detected"] == False
        out_of_bounds_mask = (
            tracking_long["x"].abs() > (float(x_limit) + float(tolerance))
        ) | (
            tracking_long["y"].abs() > (float(y_limit) + float(tolerance))
        )
        mask = undetected_mask & out_of_bounds_mask
        if mask.any():
            tracking_long.loc[mask, ["x", "y", "z"]] = pd.NA
        return tracking_long

    @staticmethod
    def load_tracking_long_data(
        tracking_path: str | Path,
        raw_metadata: dict,
        lineup: pd.DataFrame,
        match_metadata: dict,
        fps: float,
    ) -> pd.DataFrame:
        player_to_object = (
            lineup.dropna(subset=["player_id", "object_id"])
            .assign(player_id=lambda df: df["player_id"].astype("string"))
            .set_index("player_id")["object_id"]
            .astype("string")
            .to_dict()
        )

        home_team_id = lineup.loc[lineup["home_away"] == "home", "team_id"].astype("string").iloc[0]
        away_team_id = lineup.loc[lineup["home_away"] == "away", "team_id"].astype("string").iloc[0]

        period_frame_map = SkillcornerDataPreprocessor._build_period_frame_map(raw_metadata)
        if not period_frame_map:
            return pd.DataFrame()

        first_start_frame = min(
            info["start_frame"]
            for info in period_frame_map.values()
            if info.get("start_frame") is not None
        )
        kickoff_time = SkillcornerDataPreprocessor._normalize_kickoff_time(match_metadata.get("kickoff_time"))

        play_direction = match_metadata.get("play_direction")
        if not isinstance(play_direction, dict):
            play_direction = {}

        raw_pitch_length = pd.to_numeric(match_metadata.get("pitch_length"), errors="coerce")
        raw_pitch_width = pd.to_numeric(match_metadata.get("pitch_width"), errors="coerce")

        long_rows = []
        with Path(tracking_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue

                frame = json.loads(payload)
                period_id = frame.get("period")
                frame_id = frame.get("frame")
                if period_id is None or frame_id is None:
                    continue

                period_id = int(period_id)
                frame_id = int(frame_id)
                period_info = period_frame_map.get(period_id)
                if period_info is None or period_info.get("start_frame") is None:
                    continue

                period_name = CDF_PERIOD_MAP.get(period_id)
                flip_orientation = play_direction.get(period_name) == "rightleft"

                timestamp_seconds = SkillcornerDataPreprocessor._frame_seconds(
                    frame_id,
                    int(period_info["start_frame"]),
                    fps,
                )
                utc_offset_seconds = SkillcornerDataPreprocessor._frame_seconds(
                    frame_id,
                    int(first_start_frame),
                    fps,
                )

                possession = frame.get("possession") or {}
                ball_data = frame.get("ball_data") or {}

                frame_row = {
                    "period_id": period_id,
                    "timestamp": timestamp_seconds,
                    "frame_id": frame_id,
                    "utc_timestamp": (
                        kickoff_time + pd.to_timedelta(utc_offset_seconds, unit="s")
                        if pd.notna(kickoff_time)
                        else pd.NaT
                    ),
                    "ball_state": SkillcornerDataPreprocessor._infer_ball_state(possession, ball_data),
                    "ball_owning_team_id": SkillcornerDataPreprocessor._map_possession_group_to_team_id(
                        possession.get("group"),
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                    ),
                }

                ball_x = ball_data.get("x")
                ball_y = ball_data.get("y")
                if flip_orientation:
                    ball_x = -ball_x if ball_x is not None else None
                    ball_y = -ball_y if ball_y is not None else None

                long_rows.append(
                    {
                        **frame_row,
                        "player_id": pd.NA,
                        "object_id": "ball",
                        "ball": True,
                        "x": SkillcornerDataPreprocessor._rescale_tracking_coordinate(
                            ball_x,
                            raw_pitch_size=raw_pitch_length,
                            target_pitch_size=PITCH_X,
                        ),
                        "y": SkillcornerDataPreprocessor._rescale_tracking_coordinate(
                            ball_y,
                            raw_pitch_size=raw_pitch_width,
                            target_pitch_size=PITCH_Y,
                        ),
                        "z": ball_data.get("z", pd.NA),
                        "is_detected": ball_data.get("is_detected", pd.NA),
                    }
                )

                for player in frame.get("player_data") or []:
                    player_id = player.get("player_id")
                    if player_id is None:
                        continue

                    object_id = player_to_object.get(str(player_id))
                    if object_id is None or pd.isna(object_id):
                        continue

                    player_x = player.get("x")
                    player_y = player.get("y")
                    if flip_orientation:
                        player_x = -player_x if player_x is not None else None
                        player_y = -player_y if player_y is not None else None

                    long_rows.append(
                        {
                            **frame_row,
                            "player_id": str(player_id),
                            "object_id": object_id,
                            "ball": False,
                            "x": SkillcornerDataPreprocessor._rescale_tracking_coordinate(
                                player_x,
                                raw_pitch_size=raw_pitch_length,
                                target_pitch_size=PITCH_X,
                            ),
                            "y": SkillcornerDataPreprocessor._rescale_tracking_coordinate(
                                player_y,
                                raw_pitch_size=raw_pitch_width,
                                target_pitch_size=PITCH_Y,
                            ),
                            "z": pd.NA,
                            "is_detected": player.get("is_detected", pd.NA),
                        }
                    )

        tracking_df = pd.DataFrame(long_rows)
        if tracking_df.empty:
            return tracking_df

        tracking_df = tracking_df.sort_values(
            by=["period_id", "frame_id", "ball", "object_id"],
            kind="mergesort",
            ignore_index=True,
        )
        numeric_cols = [
            col for col in ("timestamp", "x", "y", "z")
            if col in tracking_df.columns
        ]
        for column in numeric_cols:
            tracking_df[column] = pd.to_numeric(tracking_df[column], errors="coerce")

        tracking_df = SkillcornerDataPreprocessor._mask_undetected_outliers(
            tracking_df,
            x_limit=PITCH_X / 2,
            y_limit=PITCH_Y / 2,
            tolerance=SkillcornerDataPreprocessor.UNDETECTED_OUTLIER_TOL,
        )

        for column in ("player_id", "object_id", "ball_owning_team_id"):
            if column in tracking_df.columns:
                tracking_df[column] = tracking_df[column].astype("string")
        if "ball" in tracking_df.columns:
            tracking_df["ball"] = tracking_df["ball"].fillna(False).astype(bool)

        return tracking_df

    @staticmethod
    def tracking_long_to_wide(tracking_long: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        if tracking_long.empty:
            return tracking_long.copy()

        time_cols = ["period_id", "timestamp", "frame_id", "utc_timestamp"]
        frame_meta_cols = time_cols + ["ball_state", "ball_owning_team_id"]
        frame_meta = tracking_long[frame_meta_cols].drop_duplicates().sort_values(
            by=["period_id", "frame_id"],
            kind="mergesort",
            ignore_index=True,
        )

        coords = tracking_long.pivot_table(
            index=time_cols,
            columns="object_id",
            values=["x", "y", "z"],
            aggfunc="first",
            sort=False,
        )
        coords.columns = [f"{object_id}_{metric}" for metric, object_id in coords.columns]
        coords = coords.reset_index(drop=False)

        tracking_df = frame_meta.merge(coords, how="left", on=time_cols)
        numeric_cols = [
            col for col in tracking_df.columns
            if col.endswith(("_x", "_y", "_z")) or col in {"timestamp"}
        ]
        for column in numeric_cols:
            tracking_df[column] = pd.to_numeric(tracking_df[column], errors="coerce")

        return SkillcornerDataPreprocessor._order_tracking_columns(tracking_df, lineup)

    @staticmethod
    def load_tracking_data(
        tracking_path: str | Path,
        raw_metadata: dict,
        lineup: pd.DataFrame,
        match_metadata: dict,
        fps: float,
    ) -> pd.DataFrame:
        tracking_long = SkillcornerDataPreprocessor.load_tracking_long_data(
            tracking_path,
            raw_metadata,
            lineup,
            match_metadata,
            fps,
        )
        return SkillcornerDataPreprocessor.tracking_long_to_wide(tracking_long, lineup)

    def _ensure_tracking_loaded(self) -> None:
        if not self.tracking.empty:
            return
        self.tracking_long = self.load_tracking_long_data(
            self.tracking_path,
            self.raw_metadata,
            self.lineup,
            self.match_metadata,
            self.fps,
        )
        self.tracking = self.tracking_long_to_wide(self.tracking_long, self.lineup)

    def preprocess_event_data(self) -> pd.DataFrame:
        raise NotImplementedError("SkillCorner event preprocessing is not implemented yet.")

    def preprocess_tracking_data(self, apply_kinematic_correction: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._ensure_tracking_loaded()
        return BaseEventTrackingPreprocessor.preprocess_tracking_data(
            self,
            apply_kinematic_correction=apply_kinematic_correction,
        )
