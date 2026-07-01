import json
import os
import re
import warnings
import xml.etree.ElementTree as ET
from fnmatch import fnmatch
from typing import Optional

import pandas as pd
from kloppy import sportec
from kloppy.domain import Dimension, MetricPitchDimensions, Orientation, TrackingDataset

from .base import BaseEventTrackingPreprocessor
from .constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y, POSITION_MAPPING

META_DIR = "data/sportec/metadata"
EVENT_DIR = "data/sportec/event"
TRACKING_DIR = "data/sportec/tracking"


class SportecDataPreprocessor(BaseEventTrackingPreprocessor):
    def __init__(self, root_dir: str, match_id: str, load_tracking: bool = True):
        super().__init__()
        self.match_id = match_id
        match_path = os.path.join(root_dir, match_id)
        
        meta_files = [f for f in os.listdir(match_path) if "matchinformation" in f or "Spielinformationen" in f]
        event_files = [f for f in os.listdir(match_path) if "events" in f or "Ereignisdaten-Spiel-Roh" in f]
        tracking_files = [f for f in os.listdir(match_path) if "positions" in f or "Positionsdaten-Spiel-Roh" in f]

        assert meta_files and event_files and tracking_files, f"Required files are missing in {match_path}"

        self.meta_path = f"{match_path}/{meta_files[0]}"
        self.event_path = f"{match_path}/{event_files[0]}"
        self.tracking_path = f"{match_path}/{tracking_files[0]}"
        # self.meta_path = f"{META_DIR}/{meta_files[0]}"
        # self.event_path = f"{EVENT_DIR}/{event_files[0]}"
        # self.tracking_path = f"{TRACKING_DIR}/{tracking_files[0]}"
        
        self.raw_metadata = self.load_raw_metadata(self.meta_path)
        self.match_metadata = self.extract_match_metadata(self.raw_metadata)
        self.lineup = self.load_lineup_data(self.raw_metadata, self.match_metadata)

        self.events = self.load_event_data(self.event_path)
        self.events = self.add_score_columns(self.events, self.lineup)
        self.events = self.align_event_orientations(self.lineup, self.events)
     
        # Since it often takes more than a minute to load tracking data, you can choose whether to delay loading
        if load_tracking:
            self.tracking_ds = self.load_kloppy_tracking_dataset(self.tracking_path, self.meta_path)
            self.fps = self.tracking_ds.frame_rate
            self.tracking = self.load_tracking_data(self.tracking_ds, self.lineup)
            
    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    @staticmethod
    def add_score_columns(events: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        tid_to_side = (
            lineup.drop_duplicates("team_id")
            .set_index("team_id")["home_away"]
            .to_dict()
        )

        home_score = 0
        away_score = 0
        home_scores = []
        away_scores = []

        for _, row in events.iterrows():
            home_scores.append(home_score)
            away_scores.append(away_score)

            side = tid_to_side.get(row["team_id"])

            if row["event_type"] == "Shot" and row["success"] is True:
                if side == "home":
                    home_score += 1
                elif side == "away":
                    away_score += 1

            elif row["event_type"] == "OwnGoal":
                if side == "home":
                    away_score += 1
                elif side == "away":
                    home_score += 1

        events["home_score"] = home_scores
        events["away_score"] = away_scores
        events["score"] = [f"{h}-{a}" for h, a in zip(home_scores, away_scores)]

        return events
    
    @staticmethod
    def load_raw_metadata(meta_path: str) -> ET.Element:
        return ET.parse(meta_path).getroot()

    @staticmethod
    def extract_match_metadata(raw_metadata: ET.Element) -> dict:
        match_info = raw_metadata.find(".//MatchInformation")

        general = match_info.find("General") if match_info is not None else None
        environment = match_info.find("Environment") if match_info is not None else None
        result_value = None if general is None else general.attrib.get("Result")
        final_home_score = pd.NA
        final_away_score = pd.NA
        if result_value:
            try:
                home_score, away_score = result_value.split(":")
                final_home_score = int(home_score)
                final_away_score = int(away_score)
            except (AttributeError, ValueError):
                pass

        def coalesce_id(id_value: Optional[str], name_value: Optional[str], field_name: str) -> Optional[str]:
            if id_value is None and name_value is not None:
                warnings.warn(
                    f"{field_name} missing in raw metadata; using {field_name.replace('_id', '_name')} instead",
                    stacklevel=2,
                )
                return name_value
            return id_value

        return {
            "competition_id": coalesce_id(
                None if general is None else general.attrib.get("CompetitionId"),
                None if general is None else general.attrib.get("CompetitionName"),
                "competition_id",
            ),
            "competition_name": None if general is None else general.attrib.get("CompetitionName"),
            "season_id": coalesce_id(
                None if general is None else general.attrib.get("SeasonId"),
                None if general is None else general.attrib.get("Season"),
                "season_id",
            ),
            "season_name": None if general is None else general.attrib.get("Season"),
            "match_id": None if general is None else general.attrib.get("MatchId"),
            "kickoff_time": None if general is None else general.attrib.get("KickoffTime"),
            # Play direction is not provided in the raw Sportec match metadata.
            "play_direction": pd.NA,
            "home_team_id": None if general is None else general.attrib.get("HomeTeamId"),
            "home_team_name": None if general is None else general.attrib.get("HomeTeamName"),
            "away_team_id": None if general is None else general.attrib.get("GuestTeamId"),
            "away_team_name": None if general is None else general.attrib.get("GuestTeamName"),
            "stadium_id": coalesce_id(
                None if environment is None else environment.attrib.get("StadiumId"),
                None if environment is None else environment.attrib.get("StadiumName"),
                "stadium_id",
            ),
            "stadium_name": None if environment is None else environment.attrib.get("StadiumName"),
            "pitch_length": None if environment is None else environment.attrib.get("PitchX"),
            "pitch_width": None if environment is None else environment.attrib.get("PitchY"),
            "final_home_score": final_home_score,
            "final_away_score": final_away_score,
            "final_score": result_value if result_value is not None else pd.NA,
            "vendor_name": "Sportec Solutions AG",
            "vendor_version": pd.NA,
            "cdf_version": "v1",
        }

    @staticmethod
    def load_lineup_data(raw_metadata: ET.Element, match_metadata: dict) -> pd.DataFrame:
        lineup_list = []
        for team in raw_metadata.findall(".//Team"):
            team_id = team.attrib.get("TeamId")
            team_name = team.attrib.get("TeamName")
            home_away = "away" if team.attrib.get("Role") == "guest" else "home"

            players = team.find("Players")
            if players is not None:
                for player in players.findall("Player"):
                    uniform_number = int(player.attrib.get("ShirtNumber"))
                    lineup_list.append(
                        {
                            "team_id": str(team_id) if team_id is not None else None,
                            "team_name": team_name,
                            "home_away": home_away,
                            "player_id": str(player.attrib.get("PersonId")) if player.attrib.get("PersonId") is not None else None,
                            "uniform_number": uniform_number,
                            "object_id": f"{home_away}_{uniform_number}",
                            "player_name": player.attrib.get("Shortname"),
                            "starting": player.attrib.get("Starting") == "true",
                            "playing_position": POSITION_MAPPING[player.attrib.get("PlayingPosition")],
                            "captain": player.attrib.get("TeamLeader") == "true",
                        }
                    )

        lineup_df = pd.DataFrame(lineup_list).sort_values(
            ["home_away", "uniform_number"], ignore_index=True
        )
        lineup_df["team_id"] = lineup_df["team_id"].astype("string")
        lineup_df["player_id"] = lineup_df["player_id"].astype("string")
        lineup_df["uniform_number"] = lineup_df["uniform_number"].astype("int64")

        return lineup_df

    # -------------------------------------------------------------------------
    # Raw event parser
    # -------------------------------------------------------------------------
    @staticmethod
    def load_event_data(event_path: str) -> pd.DataFrame:
        game_section_to_period_id = {
            "firstHalf": 1,
            "secondHalf": 2,
            "firstHalfExtra": 3,
            "secondHalfExtra": 4,
            "penalty": 5,
        }

        def _parse_play_or_shot(element):
            """Parse a Play/ShotAtGoal element into ``(event_type, attrs)``."""
            attrs = dict(element.attrib)

            if element.tag == "Play":
                pass_element = element.find("Pass")
                cross_element = element.find("Cross")
                sub = pass_element if pass_element is not None else cross_element
                if sub is not None:
                    event_type = "Pass" if sub.tag == "Pass" else "Cross"
                    attrs["play_detail"] = sub.tag
                    for k, v in sub.attrib.items():
                        attrs[f"sub_{k}"] = v
                else:
                    event_type = "Play"
                return event_type, attrs

            elif element.tag == "ShotAtGoal":
                event_type = "Shot"
                for sub in element:
                    if sub.tag in ("SuccessfulShot", "SavedShot", "BlockedShot",
                                "ShotWide", "ShotWoodWork", "OtherShot"):
                        attrs["shot_result_type"] = sub.tag
                        for k, v in sub.attrib.items():
                            attrs[f"sub_{k}"] = v
                        break
                return event_type, attrs

            else:  # FairPlay, FaultExecution, etc.
                return element.tag, attrs

        tree = ET.parse(event_path)
        root = tree.getroot()
        event_rows = []

        for event in root.findall(".//Event"):
            event_id = event.attrib.get("EventId")
            timestamp = event.attrib.get("EventTime")
            qualifiers = [dict(q.attrib) for q in event.findall("Qualifier")]

            # Event-level positions
            x = event.attrib.get("X-Position") or event.attrib.get("X-Source-Position")
            y = event.attrib.get("Y-Position") or event.attrib.get("Y-Source-Position")
            end_x = event.attrib.get("End-X-Position")
            end_y = event.attrib.get("End-Y-Position")
            start_frame = event.attrib.get("StartFrame")
            end_frame = event.attrib.get("EndFrame")
            calculated_frame = event.attrib.get("CalculatedFrame")
            calculated_timestamp = event.attrib.get("CalculatedTimestamp")
            x_position_from_tracking = event.attrib.get("X-PositionFromTracking")
            y_position_from_tracking = event.attrib.get("Y-PositionFromTracking")

            # Primary child element
            child = next((c for c in event if c.tag not in ["Qualifier"]), None)
            if child is None:
                continue
            raw_event_type = child.tag
            if raw_event_type == "Delete":
                continue

            # ── Base row ──
            row_data = {
                "event_id": event_id,
                "match_id": event.attrib.get("MatchId"),
                "raw_event_type": raw_event_type,
                "event_type": raw_event_type,
                "period_id": 0,
                "utc_timestamp": timestamp,
                "coordinates_x": float(x) if x else None,
                "coordinates_y": float(y) if y else None,
                "end_coordinates_x": float(end_x) if end_x else None,
                "end_coordinates_y": float(end_y) if end_y else None,
                "start_frame": int(start_frame) if start_frame else None,
                "end_frame": int(end_frame) if end_frame else None,
                "calculated_frame": int(calculated_frame) if calculated_frame else None,
                "calculated_timestamp": calculated_timestamp,
                "x_position_from_tracking": float(x_position_from_tracking) if x_position_from_tracking else None,
                "y_position_from_tracking": float(y_position_from_tracking) if y_position_from_tracking else None,
                "qualifiers": json.dumps(qualifiers, ensure_ascii=False) if qualifiers else None,
                "set_piece_type": None,
                "team_id": None,
                "player_id": None,
                "receiver_player_id": None,
                "success": None,
                "result": None,
                "body_part_type": None,
                "card_type": None,
            }
            for attr_key, attr_value in event.attrib.items():
                row_data[f"event_attr_{attr_key}"] = attr_value

            # ── Generic: capture ALL child attributes ──
            row_data.update(child.attrib)

            # ── Event-specific handling ──
            set_piece_tags = {"ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "KickOff"}

            if raw_event_type in set_piece_tags:
                row_data["set_piece_type"] = raw_event_type

                # Period from KickOff
                if raw_event_type == "KickOff":
                    gs = child.attrib.get("GameSection")
                    row_data["period_id"] = game_section_to_period_id.get(gs, 0)

                # Find inner Play / ShotAtGoal / FairPlay / FaultExecution
                play_element = child.find("Play")
                shot_element = child.find("ShotAtGoal")
                fair_play_element = child.find("FairPlay")
                fault_execution_element = child.find("FaultExecution")
                inner = (
                    play_element
                    if play_element is not None
                    else shot_element
                    if shot_element is not None
                    else fair_play_element
                    if fair_play_element is not None
                    else fault_execution_element
                )
                if inner is not None:
                    actual_type, inner_attrs = _parse_play_or_shot(inner)
                    row_data["event_type"] = actual_type
                    row_data.update(inner_attrs)

            elif raw_event_type == "FinalWhistle":
                gs = child.attrib.get("GameSection")
                row_data["period_id"] = game_section_to_period_id.get(gs, 0)

            elif raw_event_type in ("Play", "ShotAtGoal"):
                actual_type, extra_attrs = _parse_play_or_shot(child)
                row_data["event_type"] = actual_type
                row_data.update(extra_attrs)

            elif raw_event_type == "TacklingGame":
                duel_type = child.attrib.get("Type")
                if duel_type == "air":
                    row_data["event_type"] = "AerialDuel"
                elif duel_type == "ground":
                    row_data["event_type"] = "GroundDuel"

            elif raw_event_type == "BallClaiming":
                claiming_type = child.attrib.get("Type")
                type_map = {
                    "InterceptedBall": "Interception",
                    "BallClaimed": "Recovery",
                    "block": "Block",
                    "ballHeld": "BallHeld",
                }
                row_data["event_type"] = type_map.get(claiming_type, raw_event_type)

            elif raw_event_type == "OtherBallAction":
                if child.attrib.get("DefensiveClearance") == "true":
                    row_data["event_type"] = "Clearance"

            elif raw_event_type == "Caution":
                row_data["event_type"] = "Card"
                card_color = child.attrib.get("CardColor")
                card_map = {"yellow": "Yellow", "yellowRed": "YellowRed", "red": "Red",
                            "yellowPenaltyShootout": "YellowPenaltyShootout"}
                row_data["card_type"] = card_map.get(card_color)

            # ── Normalize: team_id ──
            row_data["team_id"] = (
                row_data.get("Team")
                or row_data.get("TeamFouler")
                or row_data.get("WinnerTeam")
            )

            # ── Normalize: player_id ──
            row_data["player_id"] = (
                row_data.get("Player")
                or row_data.get("Fouler")
                or row_data.get("Winner")
                or row_data.get("PlayerOut")
            )

            # ── Normalize: receiver_player_id ──
            receiver = (
                row_data.get("Recipient")
                or row_data.get("Fouled")
                or row_data.get("Loser")
                or row_data.get("PlayerIn")
            )
            # Shot results: goalkeeper/blocker as receiver
            srt = row_data.get("shot_result_type")
            if srt == "SavedShot":
                receiver = receiver or row_data.get("sub_GoalKeeper")
            elif srt == "BlockedShot":
                receiver = receiver or row_data.get("sub_Player")
            row_data["receiver_player_id"] = receiver

            # ── Normalize: success ──
            if row_data["success"] is None:
                evaluation = row_data.get("Evaluation")
                if evaluation is not None:
                    row_data["success"] = evaluation in ("successfullyCompleted", "successful")
                elif srt == "SuccessfulShot":
                    row_data["success"] = True
                elif srt in ("SavedShot", "BlockedShot", "ShotWide", "ShotWoodWork", "OtherShot"):
                    row_data["success"] = False
                elif raw_event_type == "TacklingGame":
                    row_data["success"] = True  # Winner perspective

            # ── Normalize: result ──
            if row_data["result"] is None:
                shot_result_map = {
                    "SuccessfulShot": "Goal", "SavedShot": "Saved", "BlockedShot": "Blocked",
                    "ShotWide": "OffTarget", "ShotWoodWork": "Post", "OtherShot": "Other",
                }
                if srt:
                    row_data["result"] = shot_result_map.get(srt)
                elif raw_event_type == "TacklingGame" and child.attrib.get("PossessionChange") == "true":
                    row_data["result"] = "PossessionChange"

            # ── Normalize: body_part_type ──
            type_of_shot = row_data.get("TypeOfShot")
            if type_of_shot:
                body_part_map = {"head": "Head", "leftLeg": "LeftFoot", "rightLeg": "RightFoot",
                                "upperBody": "Other", "noHeader": "Other"}
                row_data["body_part_type"] = body_part_map.get(type_of_shot)

            event_rows.append(row_data)

        events = pd.DataFrame(event_rows)
        events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
        events.sort_values("utc_timestamp", ignore_index=True, inplace=True)

        # Period propagation from KickOff/FinalWhistle boundaries
        period_ids = [pid for pid in events["period_id"].unique() if pid > 0]
        for period_id in period_ids:
            period_events = events[events["period_id"] == period_id]
            kickoff_idx = period_events[period_events["set_piece_type"] == "KickOff"].index
            whistle_idx = period_events[period_events["raw_event_type"] == "FinalWhistle"].index
            if len(kickoff_idx) > 0 and len(whistle_idx) > 0:
                events.loc[kickoff_idx[0]:whistle_idx[-1], "period_id"] = period_id

        return events

    # -------------------------------------------------------------------------
    # Raw events -> CDF
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_cdf_label(value: Optional[str]) -> Optional[str]:
        if value is None or pd.isna(value):
            return pd.NA

        normalized = str(value).strip()
        if normalized == "":
            return pd.NA

        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
        normalized = normalized.replace("-", "_").replace(" ", "_").replace("/", "_")
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized.lower() or pd.NA

    @staticmethod
    def _rotate_event_axis(values: pd.Series, pitch_size: float) -> pd.Series:
        rotated = pd.to_numeric(values, errors="coerce")
        valid = rotated.dropna()
        if valid.empty:
            return rotated
        if valid.min() >= 0 and valid.max() <= pitch_size + 1e-6:
            return pitch_size - rotated
        return -rotated

    @staticmethod
    def _center_event_axis(values: pd.Series, pitch_size: float) -> pd.Series:
        centered = pd.to_numeric(values, errors="coerce")
        valid = centered.dropna()
        if valid.empty:
            return centered
        if valid.min() >= 0 and valid.max() <= pitch_size + 1e-6:
            return centered - pitch_size / 2
        return centered

    @staticmethod
    def _align_raw_events_to_home_left(
        lineup: pd.DataFrame,
        events: pd.DataFrame,
        pitch_length: float,
        pitch_width: float,
    ) -> pd.DataFrame:
        events = events.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = set(gk_lineup.loc[gk_lineup["home_away"] == "home", "player_id"].astype("string"))
        away_gk_ids = set(gk_lineup.loc[gk_lineup["home_away"] == "away", "player_id"].astype("string"))

        player_ids = events["player_id"].astype("string")

        for period_id in events["period_id"].dropna().unique():
            period_mask = events["period_id"] == period_id
            period_events = events.loc[period_mask]
            period_player_ids = player_ids.loc[period_mask]

            home_gk_x = period_events.loc[period_player_ids.isin(home_gk_ids), "coordinates_x"]
            away_gk_x = period_events.loc[period_player_ids.isin(away_gk_ids), "coordinates_x"]

            if not home_gk_x.empty and not away_gk_x.empty and home_gk_x.mean() > away_gk_x.mean():
                events.loc[period_mask, "coordinates_x"] = SportecDataPreprocessor._rotate_event_axis(
                    period_events["coordinates_x"], pitch_length
                ).round(2)
                events.loc[period_mask, "coordinates_y"] = SportecDataPreprocessor._rotate_event_axis(
                    period_events["coordinates_y"], pitch_width
                ).round(2)

                if "end_coordinates_x" in events.columns:
                    events.loc[period_mask, "end_coordinates_x"] = SportecDataPreprocessor._rotate_event_axis(
                        period_events["end_coordinates_x"], pitch_length
                    ).round(2)
                if "end_coordinates_y" in events.columns:
                    events.loc[period_mask, "end_coordinates_y"] = SportecDataPreprocessor._rotate_event_axis(
                        period_events["end_coordinates_y"], pitch_width
                    ).round(2)

        return events

    @staticmethod
    def _infer_cdf_sub_type(row: pd.Series) -> Optional[str]:
        event_type = SportecDataPreprocessor._normalize_cdf_label(row.get("event_type"))
        set_piece = SportecDataPreprocessor._normalize_cdf_label(row.get("set_piece_type"))
        if set_piece is not pd.NA and pd.notna(set_piece):
            return set_piece

        card_type = SportecDataPreprocessor._normalize_cdf_label(row.get("card_type"))
        if card_type is not pd.NA and pd.notna(card_type):
            return card_type

        play_detail = SportecDataPreprocessor._normalize_cdf_label(row.get("play_detail"))
        if play_detail is not pd.NA and pd.notna(play_detail) and play_detail != event_type:
            return play_detail

        raw_event_type = SportecDataPreprocessor._normalize_cdf_label(row.get("raw_event_type"))
        wrapper_types = {"play", "shot_at_goal", "tackling_game", "ball_claiming", "other_ball_action"}
        if (
            raw_event_type is not pd.NA
            and pd.notna(raw_event_type)
            and raw_event_type != event_type
            and raw_event_type not in wrapper_types
        ):
            return raw_event_type

        return pd.NA

    @staticmethod
    def _infer_cdf_outcome(row: pd.Series) -> Optional[str]:
        success = row.get("success")
        if success is True:
            return "successful"
        if success is False:
            return "unsuccessful"

        neutral_event_types = {"card", "substitution", "final_whistle", "run", "nutmeg", "spectacular_play"}
        event_type = SportecDataPreprocessor._normalize_cdf_label(row.get("event_type"))
        if event_type in neutral_event_types:
            return "neutral"
        return pd.NA

    @staticmethod
    def _infer_cdf_outcome_detailed(row: pd.Series) -> Optional[str]:
        for field in ("result", "shot_result_type", "Evaluation", "FoulType"):
            normalized = SportecDataPreprocessor._normalize_cdf_label(row.get(field))
            if normalized is not pd.NA and pd.notna(normalized):
                return normalized
        return pd.NA

    @staticmethod
    def convert_raw_events_to_cdf(
        raw_events: pd.DataFrame,
        lineup: Optional[pd.DataFrame] = None,
        *,
        pitch_length: Optional[float] = None,
        pitch_width: Optional[float] = None,
        align_orientations: bool = True,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
    ) -> pd.DataFrame:
        events = raw_events.copy()

        pitch_length = float(pitch_length or PITCH_X)
        pitch_width = float(pitch_width or PITCH_Y)

        if lineup is not None and not lineup.empty:
            if pitch_length == PITCH_X and "pitch_length" in lineup.columns:
                lineup_pitch_length = pd.to_numeric(lineup["pitch_length"], errors="coerce").dropna()
                if not lineup_pitch_length.empty:
                    pitch_length = float(lineup_pitch_length.iloc[0])
            if pitch_width == PITCH_Y and "pitch_width" in lineup.columns:
                lineup_pitch_width = pd.to_numeric(lineup["pitch_width"], errors="coerce").dropna()
                if not lineup_pitch_width.empty:
                    pitch_width = float(lineup_pitch_width.iloc[0])

        events["source_coordinates_x"] = pd.to_numeric(events.get("coordinates_x"), errors="coerce")
        events["source_coordinates_y"] = pd.to_numeric(events.get("coordinates_y"), errors="coerce")
        events["source_end_coordinates_x"] = pd.to_numeric(events.get("end_coordinates_x"), errors="coerce")
        events["source_end_coordinates_y"] = pd.to_numeric(events.get("end_coordinates_y"), errors="coerce")

        if lineup is not None and align_orientations:
            events = SportecDataPreprocessor._align_raw_events_to_home_left(
                lineup,
                events,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )

        if lineup is not None and include_score and "home_score" not in events.columns:
            events = SportecDataPreprocessor.add_score_columns(events, lineup)

        events["coordinates_x"] = SportecDataPreprocessor._center_event_axis(events["coordinates_x"], pitch_length).round(2)
        events["coordinates_y"] = SportecDataPreprocessor._center_event_axis(events["coordinates_y"], pitch_width).round(2)
        events["end_coordinates_x"] = SportecDataPreprocessor._center_event_axis(
            events["end_coordinates_x"], pitch_length
        ).round(2)
        events["end_coordinates_y"] = SportecDataPreprocessor._center_event_axis(
            events["end_coordinates_y"], pitch_width
        ).round(2)

        events["period"] = events["period_id"].map(CDF_PERIOD_MAP).astype("string")
        events["type"] = events["event_type"].apply(SportecDataPreprocessor._normalize_cdf_label).astype("string")
        events["sub_type"] = events.apply(SportecDataPreprocessor._infer_cdf_sub_type, axis=1).astype("string")
        events["outcome"] = events.apply(SportecDataPreprocessor._infer_cdf_outcome, axis=1).astype("string")
        events["outcome_detailed"] = events.apply(
            SportecDataPreprocessor._infer_cdf_outcome_detailed, axis=1
        ).astype("string")
        events["body_part"] = events["body_part_type"].apply(
            SportecDataPreprocessor._normalize_cdf_label
        ).astype("string")
        events["receiver_id"] = events["receiver_player_id"].astype("string")
        events["receiver_time"] = pd.Series(pd.NA, index=events.index, dtype="string")
        events["related_event_ids"] = None
        events["is_synced"] = (
            events["calculated_frame"].notna()
            | events["start_frame"].notna()
            | events["x_position_from_tracking"].notna()
            | events["y_position_from_tracking"].notna()
        ).astype("boolean")
        events["tracking_frame_id"] = pd.to_numeric(
            events["calculated_frame"].combine_first(events["start_frame"]), errors="coerce"
        ).astype("Int64")
        events["tracking_frame_id_end"] = pd.to_numeric(events["end_frame"], errors="coerce").astype("Int64")
        events["x"] = events["coordinates_x"]
        events["y"] = events["coordinates_y"]
        events["x_end"] = events["end_coordinates_x"]
        events["y_end"] = events["end_coordinates_y"]

        for column in ("match_id", "event_id", "team_id", "player_id", "receiver_id"):
            if column in events.columns:
                events[column] = events[column].astype("string")

        if lineup is not None and not lineup.empty:
            player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
            events["object_id"] = events["player_id"].map(player_mapping).astype("string")
            events["receiver_object_id"] = events["receiver_id"].map(player_mapping).astype("string")

        core_columns = [
            "match_id",
            "event_id",
            "utc_timestamp",
            "period",
            "type",
            "sub_type",
            "outcome",
            "outcome_detailed",
            "team_id",
            "player_id",
            "receiver_id",
            "receiver_time",
            "body_part",
            "x",
            "y",
            "x_end",
            "y_end",
            "is_synced",
            "tracking_frame_id",
            "tracking_frame_id_end",
            "related_event_ids",
        ]

        spadl_helper_columns = [
            "success",
            "object_id",
            "receiver_object_id",
            "home_score",
            "away_score",
            "score",
        ]
        if "success" in events.columns:
            events["success"] = events["success"].astype("boolean")

        raw_helper_columns = [
            "period_id",
            "raw_event_type",
            "event_type",
            "set_piece_type",
            "play_detail",
            "shot_result_type",
            "card_type",
            "result",
            "qualifiers",
            "source_coordinates_x",
            "source_coordinates_y",
            "source_end_coordinates_x",
            "source_end_coordinates_y",
            "x_position_from_tracking",
            "y_position_from_tracking",
        ]

        preserved_columns = [
            column
            for column in events.columns
            if column not in core_columns and column not in spadl_helper_columns and column not in raw_helper_columns
        ]

        ordered_columns = [column for column in core_columns if column in events.columns]
        if include_spadl_helpers:
            ordered_columns.extend([column for column in spadl_helper_columns if column in events.columns])
        if preserve_raw_columns:
            ordered_columns.extend([column for column in raw_helper_columns if column in events.columns])
        if preserve_raw_columns:
            ordered_columns.extend(preserved_columns)

        return events.loc[:, ordered_columns].reset_index(drop=True)

    @staticmethod
    def to_cdf_events(
        raw_events: pd.DataFrame,
        lineup: Optional[pd.DataFrame] = None,
        *,
        pitch_length: Optional[float] = None,
        pitch_width: Optional[float] = None,
        align_orientations: bool = True,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
    ) -> pd.DataFrame:
        return SportecDataPreprocessor.convert_raw_events_to_cdf(
            raw_events,
            lineup=lineup,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            align_orientations=align_orientations,
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
        if raw_events is None:
            raw_events = self.load_event_data(self.event_path)

        return self.to_cdf_events(
            raw_events,
            lineup=self.lineup,
            pitch_length=self.match_metadata.get("pitch_length"),
            pitch_width=self.match_metadata.get("pitch_width"),
            include_spadl_helpers=include_spadl_helpers,
            preserve_raw_columns=preserve_raw_columns,
        )

    # -------------------------------------------------------------------------
    # CDF -> SPADL
    # -------------------------------------------------------------------------
    @staticmethod
    def infer_spadl_types(
        events_cdf: pd.DataFrame,
        *,
        include_extended_actions: bool = True,
        include_missing_actions: bool = False,
    ) -> pd.DataFrame:
        events = events_cdf.copy().reset_index(drop=True)
        events["spadl_type"] = pd.NA

        # These three action types are not part of the official socceraction
        # SPADL vocabulary. We keep them by default because Sportec's raw feed
        # contains enough contextual signal to recover useful defensive events
        # that would otherwise be dropped:
        # - ball_recovery
        # - dispossessed
        # - shot_block
        #
        # This makes the internal action table more informative for downstream
        # analysis and synchronization, while still allowing a strict SPADL-like
        # view when include_extended_actions=False.
        extended_action_types = {"ball_recovery", "dispossessed", "shot_block"}
        # Sportec's current default conversion still misses a subset of official
        # SPADL action types. The most realistic candidates to recover directly
        # from raw event semantics are:
        # - take_on
        # - keeper_save / keeper_punch
        # - keeper_claim (cross claims only, conservatively)
        # - dribble (synthetic carry inserted between adjacent actions)
        # The remaining official SPADL types that are still missing by default
        # are keeper_pick_up and the
        # Bepro-specific placeholder non_action.
        #
        # We keep this recovery behind a flag because it changes the effective
        # action vocabulary and, for keeper events, may add derived rows.

        def equal_notna(left, right) -> bool:
            return pd.notna(left) and pd.notna(right) and left == right

        def is_false(value) -> bool:
            return pd.notna(value) and bool(value) is False

        def is_interception_like(value) -> bool:
            return pd.notna(value) and value in ["interception", "clearance"]

        pass_mask = events["type"] == "pass"
        events.loc[pass_mask, "spadl_type"] = "pass"
        events.loc[pass_mask & (events["sub_type"] == "throw_in"), "spadl_type"] = "throw_in"
        events.loc[pass_mask & (events["sub_type"] == "goal_kick"), "spadl_type"] = "goalkick"
        events.loc[pass_mask & (events["sub_type"] == "corner_kick"), "spadl_type"] = "corner_short"
        events.loc[pass_mask & (events["sub_type"] == "free_kick"), "spadl_type"] = "freekick_short"

        cross_mask = events["type"] == "cross"
        events.loc[cross_mask, "spadl_type"] = "cross"
        events.loc[cross_mask & (events["sub_type"] == "corner_kick"), "spadl_type"] = "corner_crossed"
        events.loc[cross_mask & (events["sub_type"] == "free_kick"), "spadl_type"] = "freekick_crossed"

        shot_mask = events["type"] == "shot"
        events.loc[shot_mask, "spadl_type"] = "shot"
        events.loc[shot_mask & (events["sub_type"] == "free_kick"), "spadl_type"] = "shot_freekick"
        events.loc[shot_mask & (events["sub_type"] == "penalty"), "spadl_type"] = "shot_penalty"

        if include_missing_actions:
            winner_result = events.get("WinnerResult", pd.Series(pd.NA, index=events.index)).astype("string")
            dribble_evaluation = events.get(
                "DribbleEvaluation", pd.Series(pd.NA, index=events.index)
            ).astype("string")
            take_on_mask = (
                (events["type"] == "ground_duel")
                & (winner_result == "dribbledAround")
                & (dribble_evaluation == "successful")
            )
            events.loc[take_on_mask, "spadl_type"] = "take_on"

        events.loc[events["type"] == "interception", "spadl_type"] = "interception"
        events.loc[events["type"] == "recovery", "spadl_type"] = "ball_recovery"
        events.loc[events["type"] == "clearance", "spadl_type"] = "clearance"
        events.loc[events["type"] == "foul", "spadl_type"] = "foul"

        for i in events[events["type"] == "other_ball_action"].index:
            if i == 0:
                continue

            team_id = events.at[i, "team_id"]
            player_id = events.at[i, "player_id"]
            prior_non_duels = events[~events["type"].str.contains("duel", na=False)].loc[: i - 1]
            if prior_non_duels.empty:
                continue
            recent_action = prior_non_duels.iloc[-1]

            before_a_duel = (
                i + 1 in events.index
                and equal_notna(events.at[i + 1, "player_id"], player_id)
                and events.at[i + 1, "type"] == "aerial_duel"
            )
            after_a_duel = (
                i - 1 in events.index
                and equal_notna(events.at[i - 1, "player_id"], player_id)
                and events.at[i - 1, "type"] == "aerial_duel"
            )
            aerial_duel = before_a_duel or after_a_duel

            if equal_notna(recent_action["receiver_id"], player_id):
                if recent_action["type"] in ["pass", "cross"] and is_false(recent_action["success"]):
                    events.at[i, "spadl_type"] = "clearance" if aerial_duel else "interception"
                    continue
                if recent_action["type"] == "shot" and is_false(recent_action["success"]):
                    events.at[i, "spadl_type"] = "shot_block"
                    continue

            if recent_action["type"] == "clearance":
                events.at[i, "spadl_type"] = "ball_recovery"

            if i + 1 in events.index and events.at[i + 1, "type"] == "ground_duel":
                duel_winner_id = events.at[i + 1, "player_id"]
                duel_loser_id = events.at[i + 1, "receiver_id"]
                prev_player_id = events.at[i - 1, "player_id"]
                prev_event_type = events.at[i - 1, "type"]

                if equal_notna(duel_winner_id, player_id):
                    if prev_event_type == "other_ball_action" and equal_notna(duel_loser_id, prev_player_id):
                        events.at[i - 1, "spadl_type"] = "dispossessed"
                        events.at[i, "spadl_type"] = "tackle"
                        continue
                    if pd.notna(recent_action["team_id"]) and pd.notna(team_id) and recent_action["team_id"] != team_id:
                        if is_interception_like(recent_action["spadl_type"]):
                            events.at[i, "spadl_type"] = "interception"
                        else:
                            events.at[i, "spadl_type"] = "tackle"
                        continue

                if equal_notna(duel_loser_id, player_id):
                    events.at[i, "spadl_type"] = "dispossessed"
                    if prev_event_type == "other_ball_action" and equal_notna(duel_winner_id, prev_player_id):
                        events.at[i - 1, "spadl_type"] = "tackle"
                    continue

            if i - 1 in events.index and events.at[i - 1, "type"] == "ground_duel":
                duel_winner_id = events.at[i - 1, "player_id"]
                duel_loser_id = events.at[i - 1, "receiver_id"]
                if equal_notna(duel_winner_id, player_id) or equal_notna(duel_loser_id, player_id):
                    if is_interception_like(recent_action["spadl_type"]):
                        events.at[i, "spadl_type"] = "interception"
                    else:
                        events.at[i, "spadl_type"] = "tackle"
                    continue

            adj_duels = events[events["type"] == "ground_duel"].loc[max(0, i - 1): i + 2]
            involved_ids = adj_duels["player_id"].tolist() + adj_duels["receiver_id"].tolist()
            if player_id not in involved_ids and i + 1 in events.index:
                if not equal_notna(events.at[i + 1, "player_id"], player_id):
                    events.at[i, "spadl_type"] = "bad_touch"
                    continue

        if not include_extended_actions:
            events.loc[events["spadl_type"].isin(extended_action_types), "spadl_type"] = pd.NA

        always_success = ["interception", "tackle", "dispossessed", "ball_recovery", "shot_block"]
        always_failure = ["foul"]
        receiver_dependent = ["clearance", "bad_touch"]

        events.loc[events["spadl_type"].isin(always_success), "success"] = True
        events.loc[events["spadl_type"].isin(always_failure), "success"] = False

        dependent_events = events[events["spadl_type"].isin(receiver_dependent)].copy()
        spadl_events = events[events["spadl_type"].notna()].copy()

        for i in dependent_events.index:
            if i == spadl_events.index[-1]:
                events.at[i, "success"] = False
            else:
                period = events.at[i, "period"]
                team_id = events.at[i, "team_id"]
                next_event = spadl_events.loc[i + 1 :].iloc[0]
                events.at[i, "success"] = next_event["period"] == period and next_event["team_id"] == team_id

        return events

    @staticmethod
    def _infer_spadl_bodypart_name(events: pd.DataFrame) -> pd.Series:
        bodypart_map = {
            "left_foot": "foot_left",
            "right_foot": "foot_right",
            "head": "head",
            "other": "other",
        }
        return events["body_part"].map(bodypart_map).astype("string")

    @staticmethod
    def _infer_spadl_result_name(events: pd.DataFrame) -> pd.Series:
        result_name = pd.Series(pd.NA, index=events.index, dtype="string")
        result_name.loc[events["success"] == True] = "success"
        result_name.loc[events["success"] == False] = "fail"

        result_name.loc[events["type"] == "offside"] = "offside"
        result_name.loc[events["type"] == "own_goal"] = "owngoal"
        result_name.loc[events["sub_type"] == "yellow"] = "yellow_card"
        result_name.loc[events["sub_type"].isin(["red", "yellow_red"])] = "red_card"
        return result_name

    @staticmethod
    def _enrich_spadl_end_coordinates(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["end_x"] = pd.to_numeric(events["x_end"], errors="coerce")
        events["end_y"] = pd.to_numeric(events["y_end"], errors="coerce")

        pass_like = {
            "pass",
            "cross",
            "throw_in",
            "freekick_short",
            "freekick_crossed",
            "corner_short",
            "corner_crossed",
            "goalkick",
        }

        for i, row in events[events["spadl_type"].isin(pass_like)].iterrows():
            if pd.notna(row["end_x"]) and pd.notna(row["end_y"]):
                continue
            receiver_id = row.get("receiver_id")
            if pd.isna(receiver_id):
                continue
            receiver_events = events.loc[
                (events.index > i)
                & (events["period"] == row["period"])
                & (events["player_id"] == receiver_id),
                ["x", "y"],
            ]
            if not receiver_events.empty:
                events.at[i, "end_x"] = receiver_events.iloc[0]["x"]
                events.at[i, "end_y"] = receiver_events.iloc[0]["y"]

        events["end_x"] = events["end_x"].fillna(events["x"])
        events["end_y"] = events["end_y"].fillna(events["y"])
        return events

    @staticmethod
    def _copy_spadl_end_coordinates(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["end_x"] = pd.to_numeric(events["x_end"], errors="coerce")
        events["end_y"] = pd.to_numeric(events["y_end"], errors="coerce")
        return events

    @staticmethod
    def _append_missing_spadl_actions(
        events: pd.DataFrame,
        lineup: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        events = events.copy()
        events["_spadl_order"] = pd.Series(range(len(events)), index=events.index, dtype="float64")

        if lineup is None or lineup.empty:
            return events

        player_to_team = lineup.drop_duplicates("player_id").set_index("player_id")["team_id"].to_dict()
        player_to_object = lineup.drop_duplicates("player_id").set_index("player_id")["object_id"].to_dict()
        goalkeeper_ids = set(
            lineup.loc[lineup["playing_position"] == "GK", "player_id"].dropna().astype("string")
        )

        ball_claiming_types = events.get("Type", pd.Series(pd.NA, index=events.index)).astype("string")
        raw_event_types = events.get("raw_event_type", pd.Series(pd.NA, index=events.index)).astype("string")

        keeper_claim_mask = (
            events["player_id"].astype("string").isin(goalkeeper_ids)
            & events["spadl_type"].isin(["interception", "ball_recovery"])
            & (raw_event_types == "BallClaiming")
            & ball_claiming_types.isin(["BallClaimed", "BallHeld"])
        )
        for idx in events.index[keeper_claim_mask]:
            previous_spadl = events.loc[: idx - 1]
            if previous_spadl.empty:
                continue
            previous_relevant = previous_spadl[previous_spadl["spadl_type"].notna()]
            if previous_relevant.empty:
                continue
            recent_action = previous_relevant.iloc[-1]
            if (
                recent_action["spadl_type"] in ["cross", "corner_crossed", "freekick_crossed"]
                and pd.notna(recent_action["team_id"])
                and pd.notna(events.at[idx, "team_id"])
                and recent_action["team_id"] != events.at[idx, "team_id"]
            ):
                events.at[idx, "spadl_type"] = "keeper_claim"
                events.at[idx, "success"] = True
                events.at[idx, "receiver_id"] = pd.NA
                if "receiver_object_id" in events.columns:
                    events.at[idx, "receiver_object_id"] = pd.NA
                if "body_part" in events.columns:
                    events.at[idx, "body_part"] = "other"

        keeper_save_mask = (
            (events["type"] == "shot")
            & (events["outcome_detailed"] == "saved")
            & events["receiver_id"].notna()
        )
        if not keeper_save_mask.any():
            return events

        extra_rows = []
        for idx, row in events.loc[keeper_save_mask].iterrows():
            goalkeeper_id = row["receiver_id"]
            goalkeeper_team_id = player_to_team.get(goalkeeper_id)
            goalkeeper_object_id = player_to_object.get(goalkeeper_id)

            if goalkeeper_team_id is None or goalkeeper_object_id is None:
                continue

            keeper_x = row["x_end"] if pd.notna(row["x_end"]) else row["x"]
            keeper_y = row["y_end"] if pd.notna(row["y_end"]) else row["y"]

            future_spadl = events.loc[idx + 1 :]
            future_spadl = future_spadl[
                future_spadl["spadl_type"].notna()
                & (future_spadl["spadl_type"] != "shot_block")
            ]
            next_action_player_id = future_spadl.iloc[0]["player_id"] if not future_spadl.empty else pd.NA
            keeper_action_type = (
                "keeper_save"
                if pd.notna(next_action_player_id) and next_action_player_id == goalkeeper_id
                else "keeper_punch"
            )

            extra = row.copy()
            extra["team_id"] = goalkeeper_team_id
            extra["player_id"] = goalkeeper_id
            extra["object_id"] = goalkeeper_object_id
            extra["receiver_id"] = pd.NA
            extra["receiver_object_id"] = pd.NA
            extra["spadl_type"] = keeper_action_type
            extra["success"] = True
            extra["body_part"] = "other"
            extra["x"] = keeper_x
            extra["y"] = keeper_y
            extra["x_end"] = keeper_x
            extra["y_end"] = keeper_y
            extra["_spadl_order"] = float(events.at[idx, "_spadl_order"]) + 0.1
            extra_rows.append(extra)

        if not extra_rows:
            return events

        extra_df = pd.DataFrame(extra_rows).reindex(columns=events.columns)
        extra_df = extra_df.dropna(axis=1, how="all")
        events = pd.concat([events, extra_df], ignore_index=True, sort=False)
        events = events.sort_values("_spadl_order", kind="stable").reset_index(drop=True)

        goalkeeper_shot_block_indices = []
        for idx in events.index[events["spadl_type"] == "shot_block"]:
            player_id = events.at[idx, "player_id"]
            if player_id not in goalkeeper_ids:
                continue

            previous_rows = events.loc[: idx - 1]
            if previous_rows.empty:
                continue

            previous_relevant = previous_rows[
                previous_rows["spadl_type"].isin(["keeper_save", "keeper_punch", "shot"])
            ]
            if previous_relevant.empty:
                continue

            recent_action = previous_relevant.iloc[-1]
            if recent_action["spadl_type"] in ["keeper_save", "keeper_punch"] and recent_action["player_id"] == player_id:
                goalkeeper_shot_block_indices.append(idx)
                continue

            if (
                recent_action["spadl_type"] == "shot"
                and recent_action.get("outcome_detailed") == "saved"
                and recent_action.get("receiver_id") == player_id
            ):
                goalkeeper_shot_block_indices.append(idx)

        if goalkeeper_shot_block_indices:
            events = events.drop(index=goalkeeper_shot_block_indices).reset_index(drop=True)

        return events

    @staticmethod
    def _append_synthetic_dribbles(
        events: pd.DataFrame,
        *,
        min_dribble_length: float = 3.0,
        max_dribble_length: float = 60.0,
        max_dribble_duration: float = 10.0,
    ) -> pd.DataFrame:
        events = events.copy()
        if events.empty or "_spadl_order" not in events.columns:
            return events

        period_col = "period_id" if "period_id" in events.columns else "period"
        start_x_col = "start_x" if "start_x" in events.columns else "x"
        start_y_col = "start_y" if "start_y" in events.columns else "y"
        next_events = events.groupby(["match_id", period_col], sort=False).shift(-1)

        same_team = events["team_id"] == next_events["team_id"]
        next_type = next_events["spadl_type"].astype("string")
        next_bodypart = next_events.get(
            "bodypart_name", pd.Series(pd.NA, index=events.index, dtype="string")
        ).astype("string")

        not_offensive_foul = same_team & (next_type != "foul")
        not_headed_shot = ~((next_type == "shot") & (next_bodypart == "head"))
        not_bad_touch = next_type != "bad_touch"
        not_dribble = ~next_type.isin(["dribble", "take_on"])

        dx = events["end_x"] - next_events[start_x_col]
        dy = events["end_y"] - next_events[start_y_col]
        dist_sq = dx**2 + dy**2
        far_enough = dist_sq >= min_dribble_length**2
        not_too_far = dist_sq <= max_dribble_length**2

        dt = (next_events["utc_timestamp"] - events["utc_timestamp"]).dt.total_seconds()
        same_phase = dt.ge(0) & dt.lt(max_dribble_duration)

        dribble_idx = (
            same_team.fillna(False)
            & far_enough.fillna(False)
            & not_too_far.fillna(False)
            & same_phase.fillna(False)
            & not_offensive_foul.fillna(False)
            & not_headed_shot.fillna(False)
            & not_bad_touch.fillna(False)
            & not_dribble.fillna(False)
        )

        if not dribble_idx.any():
            return events

        prev = events.loc[dribble_idx].copy()
        nex = next_events.loc[dribble_idx].copy()

        dribbles = pd.DataFrame(index=prev.index)
        dribbles["match_id"] = prev["match_id"].values
        dribbles["original_event_id"] = pd.NA
        dribbles[period_col] = prev[period_col].values
        dribbles["utc_timestamp"] = (
            prev["utc_timestamp"] + (nex["utc_timestamp"] - prev["utc_timestamp"]) / 2
        ).values
        dribbles["team_id"] = nex["team_id"].values
        dribbles["player_id"] = nex["player_id"].values
        dribbles["object_id"] = nex["object_id"].values
        if "receiver_id" in events.columns:
            dribbles["receiver_id"] = pd.NA
        if "receiver_object_id" in events.columns:
            dribbles["receiver_object_id"] = pd.NA
        dribbles["spadl_type"] = "dribble"
        dribbles[start_x_col] = prev["end_x"].values
        dribbles[start_y_col] = prev["end_y"].values
        dribbles["end_x"] = nex[start_x_col].values
        dribbles["end_y"] = nex[start_y_col].values
        dribbles["bodypart_name"] = "foot"
        dribbles["result_name"] = "success"
        dribbles["success"] = True
        if "home_score" in events.columns:
            dribbles["home_score"] = prev["home_score"].values
        if "away_score" in events.columns:
            dribbles["away_score"] = prev["away_score"].values
        if "score" in events.columns:
            dribbles["score"] = prev["score"].values
        if "cdf_type" in events.columns:
            dribbles["cdf_type"] = pd.NA
        if "cdf_sub_type" in events.columns:
            dribbles["cdf_sub_type"] = pd.NA
        if "cdf_outcome_detailed" in events.columns:
            dribbles["cdf_outcome_detailed"] = pd.NA
        dribbles["_spadl_order"] = (
            prev["_spadl_order"].astype(float) + nex["_spadl_order"].astype(float)
        ) / 2
        dribbles = dribbles.dropna(axis=1, how="all")

        events = pd.concat([events, dribbles], ignore_index=True, sort=False)
        events = events.sort_values("_spadl_order", kind="stable").reset_index(drop=True)
        return events

    @staticmethod
    def to_spadl_events(
        events_cdf: pd.DataFrame,
        *,
        infer_end_coordinates: bool = False,
        include_cdf_columns: bool = False,
        include_extended_actions: bool = True,
        include_missing_actions: bool = False,
        lineup: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        events = SportecDataPreprocessor.infer_spadl_types(
            events_cdf,
            include_extended_actions=include_extended_actions,
            include_missing_actions=include_missing_actions,
        )
        if include_missing_actions:
            events = SportecDataPreprocessor._append_missing_spadl_actions(events, lineup=lineup)
        events["original_event_id"] = events["event_id"].astype("string")
        if infer_end_coordinates:
            events = SportecDataPreprocessor._enrich_spadl_end_coordinates(events)
        else:
            events = SportecDataPreprocessor._copy_spadl_end_coordinates(events)
        events["bodypart_name"] = SportecDataPreprocessor._infer_spadl_bodypart_name(events)

        take_on_mask = events["spadl_type"] == "take_on"
        if take_on_mask.any():
            events.loc[take_on_mask, "receiver_id"] = pd.NA
            if "receiver_object_id" in events.columns:
                events.loc[take_on_mask, "receiver_object_id"] = pd.NA

        if "success" not in events.columns:
            events["success"] = (events["outcome"] == "successful").astype("boolean")
        events["result_name"] = SportecDataPreprocessor._infer_spadl_result_name(events)

        selected_columns = [
            "match_id",
            "original_event_id",
            "period",
            "utc_timestamp",
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
            "_spadl_order",
        ]
        if include_cdf_columns:
            selected_columns.extend(["type", "sub_type", "outcome_detailed"])

        column_mapping = {
            "match_id": "match_id",
            "original_event_id": "original_event_id",
            "period": "period_id",
            "utc_timestamp": "utc_timestamp",
            "team_id": "team_id",
            "player_id": "player_id",
            "object_id": "object_id",
            "receiver_id": "receiver_id",
            "receiver_object_id": "receiver_object_id",
            "spadl_type": "spadl_type",
            "x": "start_x",
            "y": "start_y",
            "end_x": "end_x",
            "end_y": "end_y",
            "bodypart_name": "bodypart_name",
            "result_name": "result_name",
            "success": "success",
            "home_score": "home_score",
            "away_score": "away_score",
            "score": "score",
        }
        if include_cdf_columns:
            column_mapping.update(
                {
                    "type": "cdf_type",
                    "sub_type": "cdf_sub_type",
                    "outcome_detailed": "cdf_outcome_detailed",
                }
            )

        available_columns = [column for column in selected_columns if column in events.columns]
        input_events = events.loc[events["spadl_type"].notna(), available_columns].copy().reset_index(drop=True)
        if include_missing_actions:
            input_events = SportecDataPreprocessor._append_synthetic_dribbles(input_events)
        if "_spadl_order" in input_events.columns:
            input_events = input_events.sort_values("_spadl_order", kind="stable").reset_index(drop=True)
            input_events = input_events.drop(columns="_spadl_order")
        input_events = input_events.rename(columns=column_mapping)

        if "success" in input_events.columns:
            input_events["success"] = input_events["success"].astype(bool)

        for column in (
            "match_id",
            "original_event_id",
            "team_id",
            "player_id",
            "object_id",
            "receiver_id",
            "receiver_object_id",
            "score",
            "period_id",
            "bodypart_name",
            "result_name",
        ):
            if column in input_events.columns:
                input_events[column] = input_events[column].astype("string")

        if include_cdf_columns:
            for column in ("cdf_type", "cdf_sub_type", "cdf_outcome_detailed"):
                if column in input_events.columns:
                    input_events[column] = input_events[column].astype("string")

        if "player_id" in input_events.columns:
            input_events = input_events[input_events["player_id"].notna()].reset_index(drop=True)

        return input_events
    def preprocess_spadl_events(
        self,
        events_cdf: Optional[pd.DataFrame] = None,
        *,
        infer_end_coordinates: bool = True,
        include_cdf_columns: bool = True,
        include_extended_actions: bool = True,
        include_missing_actions: bool = False,
    ) -> pd.DataFrame:
        if events_cdf is None:
            events_cdf = self.preprocess_cdf_events(
                preserve_raw_columns=include_missing_actions,
            )
        elif include_missing_actions:
            missing_helper_columns = [
                column
                for column in ("WinnerResult", "DribbleEvaluation", "raw_event_type", "Type")
                if column not in events_cdf.columns
            ]
            if missing_helper_columns:
                enriched_events_cdf = self.preprocess_cdf_events(
                    preserve_raw_columns=True,
                )
                helper_columns = ["event_id", *missing_helper_columns]
                events_cdf = events_cdf.merge(
                    enriched_events_cdf.loc[:, helper_columns],
                    on="event_id",
                    how="left",
                )

        return self.to_spadl_events(
            events_cdf,
            infer_end_coordinates=infer_end_coordinates,
            include_cdf_columns=include_cdf_columns,
            include_extended_actions=include_extended_actions,
            include_missing_actions=include_missing_actions,
            lineup=self.lineup,
        )

    def synchronize_spadl_events(
        self,
        events_spadl: Optional[pd.DataFrame] = None,
        tracking: Optional[pd.DataFrame] = None,
        *,
        apply_kinematic_correction: bool = True,
        fps: Optional[int] = None,
        args: Optional[dict] = None,
    ) -> pd.DataFrame:
        from ..synchronize import synchronize_spadl_with_tracking

        if events_spadl is None:
            events_spadl = self.preprocess_spadl_events()

        if tracking is None:
            _, tracking = self.preprocess_tracking_data(
                apply_kinematic_correction=apply_kinematic_correction
            )

        if fps is None:
            fps = int(self.fps)

        return synchronize_spadl_with_tracking(
            events_spadl,
            tracking,
            lineup=self.lineup,
            fps=fps,
            args=args,
        )

    
    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------
    @staticmethod
    def load_kloppy_tracking_dataset(tracking_path: str, meta_path: str) -> TrackingDataset:
        tracking_ds = sportec.load_tracking(
            raw_data=tracking_path,
            meta_data=meta_path,
            coordinates="sportec",
            only_alive=False,
        )

        pitch_dims = MetricPitchDimensions(
            standardized=True,
            x_dim=Dimension(-PITCH_X / 2, PITCH_X / 2),
            y_dim=Dimension(-PITCH_Y / 2, PITCH_Y / 2),
        )
        tracking_ds = tracking_ds.transform(
            to_orientation=Orientation.STATIC_HOME_AWAY,
            to_pitch_dimensions=pitch_dims,
        )

        return tracking_ds

    @staticmethod
    def load_tracking_data(
        tracking_ds: TrackingDataset, lineup: pd.DataFrame
    ) -> pd.DataFrame:

        tracking_df: pd.DataFrame = tracking_ds.to_df()

        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        column_mapping = {f"{k}_{t}": f"{v}_{t}" for k, v in player_mapping.items() for t in ["x", "y", "d", "s"]}
        tracking_df = tracking_df.rename(columns=column_mapping)

        player_x_cols = [c for c in tracking_df.columns if fnmatch(c, "home_*_x") or fnmatch(c, "away_*_x")]
        tracking_df = tracking_df.dropna(subset=player_x_cols, how="all").copy()
        tracking_df["timestamp"] = tracking_df["timestamp"].dt.total_seconds()

        return tracking_df

    @staticmethod
    def align_event_orientations(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "home", "player_id"].tolist()
        away_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "away", "player_id"].tolist()

        for period_id in events["period_id"].unique():
            period_events = events[events["period_id"] == period_id].copy()
            home_gk_x = period_events.loc[period_events["player_id"].isin(home_gk_ids), "coordinates_x"]
            away_gk_x = period_events.loc[period_events["player_id"].isin(away_gk_ids), "coordinates_x"]

            if home_gk_x.mean() > away_gk_x.mean():  # Rotate events so that the home team plays on the left side
                events.loc[period_events.index, "coordinates_x"] = (-period_events["coordinates_x"]).round(2)
                events.loc[period_events.index, "coordinates_y"] = (-period_events["coordinates_y"]).round(2)

        return events

    @staticmethod
    def find_object_ids(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        events["object_id"] = events["player_id"].map(player_mapping)
        events["receiver_id"] = events["receiver_player_id"].map(player_mapping)

        return events

    @staticmethod
    def find_spadl_event_types(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["spadl_type"] = None

        pass_mask = events["event_type"] == "Pass"
        events.loc[pass_mask, "spadl_type"] = "pass"
        events.loc[pass_mask & (events["set_piece_type"] == "ThrowIn"), "spadl_type"] = "throw_in"
        events.loc[pass_mask & (events["set_piece_type"] == "GoalKick"), "spadl_type"] = "goalkick"
        events.loc[pass_mask & (events["set_piece_type"] == "CornerKick"), "spadl_type"] = "corner_short"
        events.loc[pass_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "freekick_short"

        cross_mask = events["event_type"] == "Cross"
        events.loc[cross_mask, "spadl_type"] = "cross"
        events.loc[cross_mask & (events["set_piece_type"] == "CornerKick"), "spadl_type"] = "corner_crossed"
        events.loc[cross_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "freekick_crossed"

        shot_mask = events["event_type"] == "Shot"
        events.loc[shot_mask, "spadl_type"] = "shot"
        events.loc[shot_mask & (events["set_piece_type"] == "FreeKick"), "spadl_type"] = "shot_freekick"
        events.loc[shot_mask & (events["set_piece_type"] == "Penalty"), "spadl_type"] = "shot_penalty"

        events.loc[events["event_type"] == "Interception", "spadl_type"] = "interception"
        events.loc[events["event_type"] == "Recovery", "spadl_type"] = "ball_recovery"
        events.loc[events["event_type"] == "Clearance", "spadl_type"] = "clearance"
        events.loc[events["event_type"] == "Foul", "spadl_type"] = "foul"

        for i in events[events["event_type"] == "OtherBallAction"].index:
            team_id = events.at[i, "team_id"]
            player_id = events.at[i, "player_id"]
            recent_action = events[~events["event_type"].str.contains("Duel", na=False)].loc[: i - 1].iloc[-1]

            before_a_duel = events.loc[i + 1, ["player_id", "event_type"]] == [player_id, "AerialDuel"]
            after_a_duel = events.loc[i - 1, ["player_id", "event_type"]] == [player_id, "AerialDuel"]
            aerial_duel = before_a_duel.all() or after_a_duel.all()

            if recent_action["receiver_player_id"] == player_id:
                if recent_action["event_type"] in ["Pass", "Cross"] and not recent_action["success"]:
                    events.at[i, "spadl_type"] = "clearance" if aerial_duel else "interception"
                    continue

                elif recent_action["event_type"] == "Shot" and not recent_action["success"]:
                    events.at[i, "spadl_type"] = "shot_block"
                    continue

            if recent_action["event_type"] == "Clearance":
                events.at[i, "spadl_type"] = "ball_recovery"

            if events.at[i + 1, "event_type"] == "GroundDuel":
                duel_winner_id = events.at[i + 1, "player_id"]
                duel_loser_id = events.at[i + 1, "receiver_player_id"]
                prev_player_id = events.at[i - 1, "player_id"]
                prev_event_type = events.at[i - 1, "event_type"]

                # If the player is the winner of the following ground duel
                if duel_winner_id == player_id:
                    if prev_event_type == "OtherBallAction" and duel_loser_id == prev_player_id:
                        events.at[i - 1, "spadl_type"] = "dispossessed"
                        events.at[i, "spadl_type"] = "tackle"
                        continue

                    elif recent_action["team_id"] != team_id:
                        if recent_action["spadl_type"] in ["interception", "clearance"]:
                            events.at[i, "spadl_type"] = "interception"
                        else:
                            events.at[i, "spadl_type"] = "tackle"
                        continue

                # If the player is the loser of the following ground duel
                if duel_loser_id == player_id:
                    events.at[i, "spadl_type"] = "dispossessed"
                    if prev_event_type == "OtherBallAction" and duel_winner_id == prev_player_id:
                        events.at[i - 1, "spadl_type"] = "tackle"
                    continue

            if events.at[i - 1, "event_type"] == "GroundDuel":
                duel_winner_id = events.at[i - 1, "player_id"]
                duel_loser_id = events.at[i - 1, "receiver_player_id"]

                # If the player is the winner of the previous ground duel
                if duel_winner_id == player_id or duel_loser_id == player_id:
                    if recent_action["spadl_type"] in ["interception", "clearance"]:
                        events.at[i, "spadl_type"] = "interception"
                    else:
                        events.at[i, "spadl_type"] = "tackle"
                    continue

            # If the player is not involved in adjoining ground duels and he/she loses possession
            adj_duels = events[events["event_type"] == "GroundDuel"].loc[i - 1 : i + 2]
            if player_id not in adj_duels["player_id"].tolist() + adj_duels["receiver_player_id"].tolist():
                if not equal_notna(events.at[i + 1, "player_id"], player_id):
                    events.at[i, "spadl_type"] = "bad_touch"
                    continue

        always_success = ["interception", "tackle", "dispossessed", "ball_recovery", "shot_block"]
        always_failure = ["foul"]
        receiver_dependent = ["clearance", "bad_touch"]

        events.loc[events["spadl_type"].isin(always_success), "success"] = True
        events.loc[events["spadl_type"].isin(always_failure), "success"] = False

        dependent_events = events[events["spadl_type"].isin(receiver_dependent)].copy()
        spadl_events = events[events["spadl_type"].notna()].copy()

        for i in dependent_events.index:
            if i == spadl_events.index[-1]:
                events.at[i, "success"] = False

            else:
                period_id = events.at[i, "period_id"]
                team_id = events.at[i, "team_id"]
                next_event = spadl_events.loc[i + 1 :].iloc[0]
                events.at[i, "success"] = next_event["period_id"] == period_id and next_event["team_id"] == team_id

        return events

    def preprocess_event_data(self) -> pd.DataFrame:
        events = SportecDataPreprocessor.find_object_ids(self.lineup, self.events)
        events = SportecDataPreprocessor.find_spadl_event_types(events)

        selected_columns = [
        "period_id",
        "utc_timestamp",
        "team_id",
        "player_id",
        "object_id",
        "spadl_type",
        "coordinates_x",
        "coordinates_y",
        "success",
        "home_score",
        "away_score",
        "score",
        ]

        column_mapping = {
            "period_id": "period_id",
            "utc_timestamp": "utc_timestamp",
            "player_id": "player_id",
            "object_id": "object_id",
            "spadl_type": "spadl_type",
            "coordinates_x": "start_x",
            "coordinates_y": "start_y",
            "success": "success",
            "home_score": "home_score",
            "away_score": "away_score",
            "score": "score",
        }
        input_events = events.loc[events["spadl_type"].notna(), selected_columns].copy().reset_index(drop=True)
        input_events = input_events.rename(columns=column_mapping).astype({"success": bool})

        input_events["team_id"] = input_events["team_id"].astype("string")
        input_events["player_id"] = input_events["player_id"].astype("string")
        input_events["object_id"] = input_events["object_id"].astype("string")

        input_events = input_events[input_events["player_id"].notna()].reset_index(drop=True)
        input_events["period_id"] = input_events["period_id"].map(CDF_PERIOD_MAP)

        return input_events

    @staticmethod
    def merge_events_and_tracking(
        lineup: pd.DataFrame,
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        fps=25,
        ffill=False,
    ) -> pd.DataFrame:
        events = events.copy()

        if "timestamp" not in events.columns:
            events = BaseEventTrackingPreprocessor.calculate_event_seconds(events)

        if "object_id" not in events.columns:
            events = SportecDataPreprocessor.find_object_ids(lineup, events)

        if "spadl_type" not in events.columns:
            events = SportecDataPreprocessor.find_spadl_event_types(events)

        return BaseEventTrackingPreprocessor.merge_events_and_tracking(events, tracking, fps, ffill)
