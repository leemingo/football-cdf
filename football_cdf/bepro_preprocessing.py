import os
import re
import json
from typing import List, Dict, Optional
import warnings

import numpy as np
import pandas as pd
from .base import BaseEventTrackingPreprocessor
from .bepro_actions import convert_to_actions
from .constants import CDF_PERIOD_MAP, PITCH_X, PITCH_Y

# Pandas 2.x exposes the future behavior behind an option. It is already the
# default in Pandas 3+, where setting the option emits a deprecation warning.
if int(pd.__version__.split(".", maxsplit=1)[0]) < 3:
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except pd.errors.OptionError:
        pass


class BeproDataPreprocessor(BaseEventTrackingPreprocessor):
    DEFAULT_HALFTIME_ASSUMPTION_MINUTES = 15.0
    _PERIOD_START_MATCH_CLOCK_SECONDS = {
        1: 0.0,
        2: 45 * 60.0,
        3: 90 * 60.0,
        4: 105 * 60.0,
        5: 120 * 60.0,
    }

    def __init__(self, root_dir: str, match_id: str, load_tracking: bool = True, target_fps: int = 25,
                 version: str = "v1"):
        """
        Bepro data loader and processor.
        Args:
            root_dir: Root directory containing match data.
            match_id: Match identifier.
            load_tracking: Whether to load tracking data immediately.
            target_fps: Tracking resample target (Hz).
            version: Raw Bepro layout. ``"v1"`` is the original per-half "extract"
                format (metadata + per-half event/frame files). ``"v2"`` is the
                Google-Drive API export (``info.json`` / ``lineup.json`` /
                ``event_data.json``). Both versions produce the SAME internal
                ``self.events`` schema (with v1-format nested ``event_types``)
                so identical downstream code (``convert_to_actions`` etc.) runs
                unchanged.
        """

        super().__init__()
        self.match_id = match_id
        self.version = version
        match_path = os.path.join(root_dir, match_id)

        if version == "v2":
            self._init_v2(match_path, load_tracking, target_fps)
            return

        meta_files = [f for f in os.listdir(match_path) if "metadata" in f]
        # event_files = sorted([f for f in os.listdir(match_path) if "1st Half" in f or "2nd Half" in f])
        event_files = sorted([f for f in os.listdir(match_path) if "event" in f])
        tracking_files = sorted([f for f in os.listdir(match_path) if "1_frame_data" in f or "2_frame_data" in f])
        assert meta_files and event_files and tracking_files, f"Required files are missing in {match_path}"

        self.meta_path = f"{match_path}/{meta_files[0]}"
        self.event_path = [f"{match_path}/{event_files[0]}", f"{match_path}/{event_files[1]}"]
        self.tracking_path = [f"{match_path}/{tracking_files[0]}", f"{match_path}/{tracking_files[1]}"]

        self.raw_metadata = self.load_raw_metadata(self.meta_path)
        self.source_fps, self.ground_width, self.ground_height = self.raw_metadata["fps"], self.raw_metadata["ground_width"], self.raw_metadata["ground_height"]
        self.target_fps = target_fps  # Resample Bepro tracking to the project-wide 25 Hz target.
        self.halftime_assumption_minutes = self.DEFAULT_HALFTIME_ASSUMPTION_MINUTES
        self.match_metadata = self.extract_match_metadata(self.raw_metadata)
        self.lineup = self.load_lineup_data(self.raw_metadata, self.match_metadata)

        self.events = self.load_event_data(self.event_path)
        self.events = self.align_event_identifier(self.lineup, self.events, self.match_id)
        self.events = self.add_score_columns(self.events, self.lineup)
        self.events = self.align_event_orientations(self.lineup, self.events)

        # Since it often takes more than a minute to load tracking data, you can choose whether to delay loading
        if load_tracking:
            self.tracking = self.load_tracking_data(self.tracking_path, self.raw_metadata, self.lineup, self.target_fps)
            self.tracking = self.align_tracking_orientations(self.lineup, self.tracking)

    def _init_v2(self, match_path: str, load_tracking: bool, target_fps: int) -> None:
        """Google-Drive (v2) initialization path.

        Produces the same ``self.events`` / ``self.lineup`` / ``self.raw_metadata``
        / ``self.match_metadata`` contract as the v1 ``__init__`` so the shared
        downstream pipeline runs unchanged. Tracking is not available in this
        export, so it is skipped regardless of ``load_tracking``.
        """
        files = os.listdir(match_path)

        def _pick(substr: str) -> Optional[str]:
            hits = sorted(f for f in files if substr in f)
            return f"{match_path}/{hits[0]}" if hits else None

        self.info_path = _pick("info")
        self.lineup_path = _pick("lineup")
        self.event_path = _pick("event_data")
        assert self.info_path and self.lineup_path and self.event_path, (
            f"Required v2 files (info/lineup/event_data) are missing in {match_path}"
        )

        self.raw_metadata = self.load_metadata_v2(self.info_path)
        self.source_fps = self.raw_metadata.get("fps")
        self.ground_width = self.raw_metadata.get("ground_width")
        self.ground_height = self.raw_metadata.get("ground_height")
        self.target_fps = target_fps
        self.halftime_assumption_minutes = self.DEFAULT_HALFTIME_ASSUMPTION_MINUTES
        self.match_metadata = self.extract_match_metadata_v2(self.raw_metadata)
        home_team_id = (self.raw_metadata.get("home_team") or {}).get("team_id")
        away_team_id = (self.raw_metadata.get("away_team") or {}).get("team_id")
        self.lineup = self.load_lineup_data_v2(
            self.lineup_path,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=(self.raw_metadata.get("home_team") or {}).get("team_name"),
            away_team_name=(self.raw_metadata.get("away_team") or {}).get("team_name"),
        )

        self.events = self.load_event_data_v2(self.event_path)
        self.events = self.align_event_identifier(self.lineup, self.events, self.match_id)
        self.events = self.add_score_columns(self.events, self.lineup)
        # Drive coordinates are already attack-normalized (attacking team toward
        # y->1, i.e. +x after the axis swap in load_event_data_v2). The GK-heuristic
        # align_event_orientations expects absolute/team-fixed coordinates, so it
        # does not apply here. Reproduce the v1 home-left frame deterministically:
        # keep home as +x and flip the away team.
        self.events = self._v2_orient_home_left(self.events)

        # Tracking frames are not part of the Google-Drive export.
        if load_tracking:
            warnings.warn(
                "Tracking data is not available for the Bepro v2 (Google-Drive) "
                "format; skipping tracking load.",
                stacklevel=2,
            )
     
    @staticmethod
    def load_raw_metadata(meta_path: str) -> dict:
        with open(meta_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)
        return raw_metadata
    
    @staticmethod
    def extract_match_metadata(raw_metadata: dict) -> dict:
        def coalesce_id(id_value, name_value, field_name: str):
            if id_value is None and name_value is not None:
                warnings.warn(
                    f"{field_name} missing in raw metadata; using {field_name.replace('_id', '_name')} instead",
                    stacklevel=2,
                )
                return name_value
            return id_value

        home_team = raw_metadata.get("home_team") or {}
        away_team = raw_metadata.get("away_team") or {}
        match_result = raw_metadata.get("match_result") or {}
        final_home_score = match_result.get("home_team_score", pd.NA)
        final_away_score = match_result.get("away_team_score", pd.NA)
        final_score = pd.NA
        if pd.notna(final_home_score) and pd.notna(final_away_score):
            final_score = f"{final_home_score}:{final_away_score}"
        
        return {
            "competition_id": coalesce_id(
                raw_metadata.get("competition_id"),
                raw_metadata.get("competition_name"),
                "competition_id",
            ),
            "competition_name": raw_metadata.get("competition_name"),
            "season_id": coalesce_id(
                raw_metadata.get("season_id"),
                raw_metadata.get("season_name"),
                "season_id",
            ),
            "season_name": raw_metadata.get("season_name"),
            "match_id": str(raw_metadata.get("match_id")) if raw_metadata.get("match_id") is not None else pd.NA,
            "kickoff_time": raw_metadata.get("match_datetime"),
            "play_direction": pd.NA,
            "home_team_id": str(home_team.get("team_id")) if home_team.get("team_id") is not None else pd.NA,
            "home_team_name": home_team.get("team_name", pd.NA),
            "away_team_id": str(away_team.get("team_id")) if away_team.get("team_id") is not None else pd.NA,
            "away_team_name": away_team.get("team_name", pd.NA),
            "stadium_id": coalesce_id(
                raw_metadata.get("stadium_id"),
                raw_metadata.get("stadium_name"),
                "stadium_id",
            ),
            "stadium_name": raw_metadata.get("stadium_name"),
            "pitch_length": raw_metadata.get("ground_width"),
            "pitch_width": raw_metadata.get("ground_height"),
            "final_home_score": final_home_score,
            "final_away_score": final_away_score,
            "final_score": final_score,
            "vendor_name": "Bepro",
            "vendor_version": pd.NA,
            "cdf_version": "v1",
        }

    @staticmethod
    def _kickoff_utc(raw_metadata: dict) -> pd.Timestamp:
        kickoff = pd.to_datetime(raw_metadata.get("match_datetime"))
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        else:
            kickoff = kickoff.tz_convert("UTC")
        return kickoff.tz_localize(None)

    @classmethod
    def _period_start_utc_offsets(
        cls,
        period_ids: pd.Series,
        *,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.Series:
        numeric_periods = pd.to_numeric(period_ids, errors="coerce")
        offsets = numeric_periods.map(cls._PERIOD_START_MATCH_CLOCK_SECONDS).astype("Float64")
        offsets = offsets + numeric_periods.ge(2).astype("Float64") * float(halftime_assumption_minutes) * 60.0
        return offsets

    @classmethod
    def _approx_utc_from_period_elapsed_seconds(
        cls,
        raw_metadata: dict,
        period_ids: pd.Series,
        elapsed_seconds: pd.Series,
        *,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.Series:
        kickoff_utc = cls._kickoff_utc(raw_metadata)
        period_start_offsets = cls._period_start_utc_offsets(
            period_ids,
            halftime_assumption_minutes=halftime_assumption_minutes,
        )
        elapsed = pd.to_numeric(elapsed_seconds, errors="coerce")
        result = pd.Series(pd.NaT, index=elapsed_seconds.index, dtype="datetime64[ns]")
        valid = period_start_offsets.notna() & elapsed.notna()
        if valid.any():
            total_offsets = (
                pd.to_timedelta(period_start_offsets.loc[valid], unit="s")
                + pd.to_timedelta(elapsed.loc[valid], unit="s")
            )
            result.loc[valid] = kickoff_utc + total_offsets
        return result

    @classmethod
    def _approx_utc_from_match_clock_ms(
        cls,
        raw_metadata: dict,
        period_ids: pd.Series,
        match_clock_ms: pd.Series,
        *,
        period_start_time_ms: Optional[pd.Series] = None,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.Series:
        match_clock_seconds = pd.to_numeric(match_clock_ms, errors="coerce") / 1000.0
        if period_start_time_ms is None:
            period_start_seconds = cls._period_start_utc_offsets(
                period_ids,
                halftime_assumption_minutes=0.0,
            )
        else:
            period_start_seconds = pd.to_numeric(period_start_time_ms, errors="coerce") / 1000.0
            missing = period_start_seconds.isna()
            if missing.any():
                fallback = cls._period_start_utc_offsets(
                    period_ids,
                    halftime_assumption_minutes=0.0,
                )
                period_start_seconds.loc[missing] = fallback.loc[missing]
        elapsed_seconds = match_clock_seconds - period_start_seconds
        return cls._approx_utc_from_period_elapsed_seconds(
            raw_metadata,
            period_ids,
            elapsed_seconds,
            halftime_assumption_minutes=halftime_assumption_minutes,
        )


    @staticmethod
    def load_lineup_data(raw_metadata: dict, match_metadata: dict) -> pd.DataFrame:
        home_team_info, away_team_info = raw_metadata['home_team'], raw_metadata['away_team']
        home_team_rows = []

        base_info = {
                'team_id': home_team_info.get('team_id'),
                'team_name': home_team_info.get('team_name'),
                'home_away': 'home',
                'player_id': None, 
                'uniform_number': None,
                'object_id': None,
                'player_name': None,
                'playing_position': None,
                'starting': None,
            }
        for _, player_data in enumerate(home_team_info['players']):
            player_info = base_info.copy()
            player_info['player_name'] = player_data['full_name_en']
            player_info['playing_position'] = player_data['initial_position_name']
            # Bepro exposes the starting lineup flag in raw metadata.
            player_info['starting'] = player_data.get('is_starting')
            player_info['uniform_number'] = player_data['shirt_number']
            player_info['player_id'] = player_data['player_id']
            player_info['object_id'] = f"home_{player_data['shirt_number']}"

            home_team_rows.append(player_info)

        home_df = pd.DataFrame(home_team_rows)
        # home_df['player_id'] = home_df['player_id'].astype(int)
        # home_df['team_id'] = home_df['team_id'].astype(int)
        home_df['player_id'] = home_df['player_id'].astype("string")
        home_df['team_id'] = home_df['team_id'].astype("string")
        home_df['uniform_number'] = home_df['uniform_number'].astype("int64")

        away_team_rows = []
        base_info['home_away'] = 'away'
        base_info['team_id'] = away_team_info.get('team_id')
        base_info['team_name'] = away_team_info.get('team_name')
        for _, player_data in enumerate(away_team_info['players']):
            player_info = base_info.copy()
            player_info['player_name'] = player_data['full_name_en']
            player_info['playing_position'] = player_data['initial_position_name']
            player_info['starting'] = player_data.get('is_starting')
            player_info['uniform_number'] = player_data['shirt_number']
            player_info['player_id'] = player_data['player_id']
            player_info['object_id'] = f"away_{player_data['shirt_number']}"

            away_team_rows.append(player_info)

        away_df = pd.DataFrame(away_team_rows)
        # away_df['player_id'] = away_df['player_id'].astype(int)
        # away_df['team_id'] = away_df['team_id'].astype(int)
        away_df['player_id'] = away_df['player_id'].astype("string")
        away_df['team_id'] = away_df['team_id'].astype("string")
        away_df['uniform_number'] = away_df['uniform_number'].astype("int64")

        lineup_df = pd.concat([home_df, away_df], ignore_index=True)
        return lineup_df

    # =====================================================================
    # v2 (Google-Drive API) loaders
    # =====================================================================

    # Map the Drive ``event_period`` enum to the v1 ``period_order`` integer.
    _V2_PERIOD_ORDER: dict[str, int] = {
        "FIRST_HALF": 0,
        "SECOND_HALF": 1,
        "FIRST_HALF_EXTRA": 2,
        "SECOND_HALF_EXTRA": 3,
        "PENALTY_SHOOTOUT": 4,
    }

    # Drive Pass.outcome / Shot-like outcome → v1 ``property.Outcome``.
    _V2_PASS_OUTCOME: dict[str, str] = {
        "Successful": "Succeeded",
        "Unsuccessful": "Failed",
    }
    # Drive Shot.outcome → v1 "Shots & Goals" property.Outcome. Only "Goal"
    # yields a SPADL success; every other (non-goal) outcome resolves to a
    # "fail" result, and the specific label is discarded afterwards — so any
    # unmapped Drive outcome safely defaults to a recognized fail label below.
    _V2_SHOT_OUTCOME: dict[str, str] = {
        "Goal": "Goals",
        "On Target": "Shots On Target",
        "Off Target": "Shots Off Target",
        "Blocked": "Blocked Shots",
        "Keeper Rush-Out": "Keeper Rush-outs",
        "Low Quality Shot": "Shots Off Target",
    }
    # Drive Set Piece.sub_event_type → v1 "Set Pieces" property.Type.
    _V2_SET_PIECE_TYPE: dict[str, str] = {
        "Goal Kick": "Goal Kicks",
        "Throw-In": "Throw-Ins",
        "Corner": "Corners",
        "Freekick": "Freekicks",
        "Free Kick": "Freekicks",
        "Penalty Kick": "Penalty Kicks",
    }
    # Drive Save.sub_event_type → v1 "Saves" property.Type.
    _V2_SAVE_TYPE: dict[str, str] = {
        "Catch": "Catches",
        "Catches": "Catches",
        "Parry": "Parries",
        "Parries": "Parries",
    }
    # Drive Duel.sub_event_type → v1 "Duels" property.Type.
    _V2_DUEL_TYPE: dict[str, str] = {
        "Aerial": "Aerial Duels",
        "Ground": "Physical Duels",
        "Loose Ball": "Loose Ball Duels",
    }
    # Drive Tackle.outcome → v1 "Tackles" property.Outcome.
    _V2_TACKLE_OUTCOME: dict[str, str] = {
        "Successful": "Succeeded",
        "Unsuccessful": "Failed",
    }
    # Drive Take-On.outcome → v1 "Take-on" property.Outcome.
    _V2_TAKE_ON_OUTCOME: dict[str, str] = {
        "Successful": "Succeeded",
        "Unsuccessful": "Failed",
    }
    # Drive body_part → bodypart hint carried on the v1 property (passthrough;
    # bepro_actions currently hardcodes foot for shots so this is informational).
    _V2_BODY_PART: dict[str, str] = {
        "Left Foot": "foot_left",
        "Right Foot": "foot_right",
        "Head": "head",
        "Hands": "other",
        "Upper Body": "other",
    }
    # Simple Drive event_type → v1 event_name (single-result / no extra property).
    _V2_SIMPLE_EVENT_NAME: dict[str, str] = {
        "Interception": "Interceptions",
        "Clearance": "Clearances",
        "Aerial Clearance": "Aerial Control",
        "Recovery": "Recoveries",
        "Block": "Blocks",
        "Step-in": "Step-in",
        "Take-On": "Take-on",
        "Goal Conceded": "Goals Conceded",
        "Pass Received": "Passes Received",
        "Cross Received": "Crosses Received",
        "Offside": "Offsides",
        "Set Piece Defence": "Set Piece Defence",
    }

    @staticmethod
    def load_metadata_v2(info_path: str) -> dict:
        """Load and flatten the Drive ``info.json`` into a v1-shaped raw_metadata
        dict (same keys ``extract_match_metadata``/``load_lineup_data`` rely on,
        plus the bookkeeping fields used elsewhere)."""
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        result = info.get("result", info) or {}

        home = result.get("home_team") or {}
        away = result.get("away_team") or {}
        season = result.get("season") or {}
        venue = result.get("venue") or {}
        match_result = result.get("detail_match_result") or {}

        return {
            "match_id": result.get("id"),
            "match_title": result.get("match_title"),
            "match_datetime": result.get("start_time"),
            "match_full_time": (result.get("full_time") * 60 * 1000)
            if result.get("full_time") is not None else None,
            "match_extra_time": result.get("extra_full_time"),
            "competition_id": season.get("league_id"),
            "competition_name": season.get("season_group_name"),
            "season_id": season.get("id"),
            "season_name": season.get("name"),
            "home_team": {
                "team_id": home.get("id"),
                "team_name": home.get("name"),
                "team_name_en": home.get("name_en"),
            },
            "away_team": {
                "team_id": away.get("id"),
                "team_name": away.get("name"),
                "team_name_en": away.get("name_en"),
            },
            "match_result": {
                "home_team_score": match_result.get("home_team_score"),
                "away_team_score": match_result.get("away_team_score"),
            },
            "stadium_id": venue.get("id"),
            "stadium_name": venue.get("display_name"),
            "ground_width": venue.get("ground_width"),
            "ground_height": venue.get("ground_height"),
            # Tracking frames are absent in the Drive export.
            "fps": None,
        }

    @staticmethod
    def extract_match_metadata_v2(raw_metadata: dict) -> dict:
        """v2 counterpart to ``extract_match_metadata`` (cdf_version='v2').

        Reuses the v1 extractor (the v2 raw_metadata already mirrors the v1
        shape) and only overrides the version stamp.
        """
        meta = BeproDataPreprocessor.extract_match_metadata(raw_metadata)
        meta["cdf_version"] = "v2"
        return meta

    @staticmethod
    def load_lineup_data_v2(
        lineup_path: str,
        *,
        home_team_id=None,
        away_team_id=None,
        home_team_name=None,
        away_team_name=None,
    ) -> pd.DataFrame:
        """Build the v1-format lineup DataFrame from the Drive ``lineup.json``.

        The Drive lineup carries ``team_id`` per player but not an explicit
        home/away flag, so home/away is taken from the match metadata
        (``info.json`` home/away team ids). If those are unavailable, falls back
        to team-id ordering (first team seen = home).
        """
        with open(lineup_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        players = payload.get("result", payload) or []

        rows = []
        for p in players:
            full_name = " ".join(
                part for part in [p.get("player_last_name"), p.get("player_name")]
                if part
            ).strip() or p.get("player_name")
            rows.append({
                "team_id": p.get("team_id"),
                "team_name": None,  # assigned below from metadata
                "home_away": None,  # assigned below
                "player_id": p.get("player_id"),
                "uniform_number": p.get("back_number"),
                "object_id": None,  # assigned below
                "player_name": full_name,
                "playing_position": p.get("position_name"),
                "starting": p.get("is_starting_lineup"),
            })

        lineup_df = pd.DataFrame(rows)

        # Assign home/away from the match metadata team ids when available, else
        # fall back to team-id ordering (first team seen = home).
        if home_team_id is None:
            team_order = list(dict.fromkeys(lineup_df["team_id"].tolist()))
            home_team_id = team_order[0] if team_order else None
        lineup_df["home_away"] = lineup_df["team_id"].apply(
            lambda t: "home" if t == home_team_id else "away"
        )
        lineup_df["team_name"] = lineup_df["home_away"].map(
            {"home": home_team_name, "away": away_team_name}
        )
        # object_id uses player_id (v2 may lack reliable shirt numbers per the
        # documented contract) so it is stable for joins.
        lineup_df["object_id"] = lineup_df.apply(
            lambda r: f"{r['home_away']}_{r['player_id']}", axis=1
        )

        lineup_df["player_id"] = lineup_df["player_id"].astype("string")
        lineup_df["team_id"] = lineup_df["team_id"].astype("string")
        lineup_df["uniform_number"] = (
            pd.to_numeric(lineup_df["uniform_number"], errors="coerce")
            .astype("Int64")
        )
        return lineup_df

    @staticmethod
    def _v2_translate_event_types(event_types: list) -> list:
        """Translate a Drive ``event_types`` list into the v1 nested format
        ``[{event_name, property:{Outcome|Type, ...}}]``.

        A single Drive row can yield multiple v1 entries (e.g. a key/assist Pass
        emits "Passes" + "Key Passes" + "Assists"; a Set-Piece Pass emits
        "Passes" + "Set Pieces").
        """
        out: list[dict] = []
        cls = BeproDataPreprocessor

        # First pass: locate the optional Set Piece descriptor so it can be
        # attached to the co-listed Pass/Shot exactly as v1 records it.
        set_piece_type = None
        for et in event_types:
            if et.get("event_type") == "Set Piece":
                set_piece_type = cls._V2_SET_PIECE_TYPE.get(
                    et.get("sub_event_type"), et.get("sub_event_type")
                )

        # Drive can split a single foul into two entries (a plain "Foul" plus a
        # "Yellow Card"/"Red Card"). v1 records ONE Fouls event carrying the
        # card Type, so collapse to the card sub-type when one is present and
        # emit only a single Fouls entry.
        foul_card_sub = None
        has_plain_foul = False
        for et in event_types:
            if et.get("event_type") == "Foul":
                sub = et.get("sub_event_type")
                if sub in ("Yellow Card", "Red Card"):
                    foul_card_sub = sub
                else:
                    has_plain_foul = True
        foul_emitted = False

        for et in event_types:
            t = et.get("event_type")
            if t == "Pass":
                event_name = "Crosses" if et.get("cross") else "Passes"
                prop = {"Outcome": cls._V2_PASS_OUTCOME.get(et.get("outcome"), et.get("outcome"))}
                out.append({"event_name": event_name, "property": prop})
                if et.get("key_pass"):
                    out.append({"event_name": "Key Passes", "property": {}})
                if et.get("assist"):
                    out.append({"event_name": "Assists", "property": {}})
            elif t == "Shot":
                # Unmapped (rare) Drive shot outcomes default to a recognized
                # non-goal fail label so the full-season build never raises.
                prop = {"Outcome": cls._V2_SHOT_OUTCOME.get(et.get("outcome"), "Shots Off Target")}
                bp = et.get("body_part")
                if bp is not None:
                    prop["Body Part"] = cls._V2_BODY_PART.get(bp, "other")
                out.append({"event_name": "Shots & Goals", "property": prop})
            elif t == "Set Piece":
                # Emit the v1 "Set Pieces" descriptor as its own entry (matching
                # the v1 nested format). _get_type_name reads this co-listed
                # entry to specialize the Pass/Shot into throw_in / corner_* /
                # freekick_* / goalkick / shot_penalty etc.
                if set_piece_type is not None:
                    out.append({"event_name": "Set Pieces", "property": {"Type": set_piece_type}})
            elif t == "Tackle":
                out.append({
                    "event_name": "Tackles",
                    "property": {"Outcome": cls._V2_TACKLE_OUTCOME.get(et.get("outcome"), et.get("outcome"))},
                })
            elif t == "Take-On":
                out.append({
                    "event_name": "Take-on",
                    "property": {"Outcome": cls._V2_TAKE_ON_OUTCOME.get(et.get("outcome"), et.get("outcome"))},
                })
            elif t == "Duel":
                out.append({
                    "event_name": "Duels",
                    "property": {"Type": cls._V2_DUEL_TYPE.get(et.get("sub_event_type"), et.get("sub_event_type"))},
                })
            elif t == "Save":
                out.append({
                    "event_name": "Saves",
                    "property": {"Type": cls._V2_SAVE_TYPE.get(et.get("sub_event_type"), et.get("sub_event_type"))},
                })
            elif t == "Aerial Clearance":
                out.append({
                    "event_name": "Aerial Control",
                    "property": {"Outcome": cls._V2_PASS_OUTCOME.get(et.get("outcome"), et.get("outcome"))},
                })
            elif t == "Foul":
                sub = et.get("sub_event_type")
                foul_type = {
                    "Yellow Card": "Yellow Cards",
                    "Red Card": "Red Cards",
                }.get(sub, "Fouls")
                out.append({"event_name": "Fouls", "property": {"Type": foul_type}})
            elif t == "Foul Won":
                out.append({"event_name": "Fouls", "property": {"Type": "Fouls Won"}})
            elif t == "Error":
                out.append({"event_name": "Mistakes", "property": {}})
            elif t == "Intervention":
                # Drive splits what the v1 extract bundles as "Tackles" into
                # "Tackle" (70) + "Intervention" (55). Map Intervention to a
                # v1 "Tackles" entry so totals match. Drive carries no outcome
                # for interventions, so default to a failed tackle (matches how
                # v1 encodes these as failed tackles in validated samples).
                out.append({"event_name": "Tackles", "property": {"Outcome": "Failed"}})
            elif t in cls._V2_SIMPLE_EVENT_NAME:
                out.append({"event_name": cls._V2_SIMPLE_EVENT_NAME[t], "property": {}})
            else:
                # Unmapped Drive event types (Intervention, Substitution, Hit,
                # Ball Received, Pause, ...) carry no v1 counterpart and are
                # dropped (mirrors v1 ignoring physical-metric rows).
                continue

        return out

    def _v2_orient_home_left(self, events: pd.DataFrame) -> pd.DataFrame:
        """Orient Drive (attack-normalized) coordinates to the v1 home-left frame.

        After the axis swap in ``load_event_data_v2`` the attacking team always
        plays toward +x. To match the v1 convention (home attacks +x, away -x),
        negate the away team's coordinates. This replaces the GK-heuristic
        ``align_event_orientations`` for v2, which assumes absolute coordinates.
        """
        events = events.copy()
        # Compare team ids numerically: some seasons serialize team_id as a float
        # ("4648.0") while the metadata id is an int ("4648"), so a plain string
        # equality silently matches nothing and leaves the away team un-flipped.
        away_raw = self.match_metadata.get("away_team_id")
        away_num = pd.to_numeric(pd.Series([away_raw]), errors="coerce").iloc[0]
        if pd.notna(away_num):
            away = pd.to_numeric(events["team_id"], errors="coerce") == away_num
        else:
            away = events["team_id"].astype("string") == str(away_raw)
        for col in ["x", "to_x", "y", "to_y"]:
            if col in events.columns:
                events.loc[away, col] = -pd.to_numeric(events.loc[away, col], errors="coerce")
        return events

    @staticmethod
    def load_event_data_v2(event_path: str) -> pd.DataFrame:
        """Load the Drive ``event_data.json`` and emit the SAME internal schema
        as the v1 ``load_event_data`` (nested v1-format ``event_types`` plus all
        flat helper columns), so identical downstream code runs unchanged."""
        with open(event_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_events = payload.get("result", payload) or []

        cls = BeproDataPreprocessor
        rows = []
        for e in raw_events:
            raw_types = e.get("event_types") or []
            if not raw_types:
                # Empty event_types are junk rows (also missing team_id); skip,
                # mirroring v1 dropping rows without an event_name.
                continue
            translated = cls._v2_translate_event_types(raw_types)
            if not translated:
                continue

            rel = e.get("relative_event") or {}
            period_order = cls._V2_PERIOD_ORDER.get(e.get("event_period"))

            rows.append({
                "period_order": period_order,
                "event_time": e.get("event_time"),
                "team_name": None,
                "player_shirt_number": None,
                "player_name": None,
                "event_types": translated,
                # Drive axes are SWAPPED vs the v1 "extract" format: Drive ``x``
                # is lateral and ``y`` is goal-ward (attack-normalized — the
                # attacking team always plays toward y->1), whereas the shared
                # rescale below treats ``x`` as the 105 m length axis and ``y``
                # as the 68 m width axis (as in extract). Map Drive y->length(x)
                # and Drive x->width(y) (same for the relative-event endpoint)
                # so shots land near the goal, matching v1.
                "x": e.get("y"),
                "y": e.get("x"),
                "to_x": rel.get("y"),
                "to_y": rel.get("x"),
                "attack_direction": e.get("attack_direction"),
                # v2 identifiers are present directly on the event.
                "team_id": e.get("team_id"),
                "player_id": e.get("player_id"),
                "event_id": e.get("id"),
                # Provider-supplied xg benchmark passthrough.
                "provider_xg": e.get("xg"),
            })

        events = pd.DataFrame(rows)
        # Drop rows lacking a usable period mapping (defensive).
        events = events[events["period_order"].notna()].reset_index(drop=True)
        events["period_order"] = events["period_order"].astype("int64")

        events["period_id"] = events["period_order"] + 1
        for col in ["x", "to_x"]:
            events[col] = events[col] * PITCH_X - PITCH_X / 2
        for col in ["y", "to_y"]:
            events[col] = events[col] * PITCH_Y - PITCH_Y / 2

        # ID dtype parity with v1 (align_event_identifier expects string ids).
        events["team_id"] = events["team_id"].astype("string")
        events["player_id"] = events["player_id"].astype("string")
        events["event_id"] = events["event_id"].astype("string")

        return cls._derive_event_flat_columns(events)

    # Priority order for determining the primary event_name of a multi-event row.
    _PRIMARY_EVENT_PRIORITY: list[str] = [
        "Passes", "Crosses", "Shots & Goals", "Take-on", "Step-in",
        "Tackles", "Interceptions", "Clearances", "Fouls", "Saves",
        "Aerial Control", "Mistakes", "Own Goals",
        "Blocks", "Defensive Line Supports",
        "Duels", "Recoveries",
        "Passes Received", "Crosses Received",
        "Set Piece Defence", "Goals Conceded", "Offsides",
    ]

    @staticmethod
    def load_event_data(event_path: list[str]) -> pd.DataFrame:
        with open(event_path[0], 'r', encoding='utf-8') as f:
            first_half_event_data = json.load(f)
        with open(event_path[1], 'r', encoding='utf-8') as f:
            second_half_event_data = json.load(f)

        first_half_event_df = pd.DataFrame(first_half_event_data['data'])
        second_half_event_df = pd.DataFrame(second_half_event_data['data'])
        events = pd.concat([first_half_event_df, second_half_event_df], axis=0, ignore_index=True)

        events["period_id"] = events["period_order"] + 1
        for col in ["x", "to_x"]:
            events[col] = events[col] * PITCH_X - PITCH_X / 2
        for col in ["y", "to_y"]:
            events[col] = events[col] * PITCH_Y - PITCH_Y / 2

        # Rename raw JSON field to avoid clash with pandas internals
        events = events.rename(columns={"events": "event_types"})

        # Drop physical-metric rows (VHIR, SPRINT, HIR, MAX_SPEED)
        # These use "name" key instead of "event_name" in the raw JSON.
        has_event_name = events["event_types"].apply(
            lambda el: any(isinstance(e, dict) and "event_name" in e for e in el)
        )
        events = events[has_event_name].reset_index(drop=True)

        return BeproDataPreprocessor._derive_event_flat_columns(events)

    @staticmethod
    def _derive_event_flat_columns(events: pd.DataFrame) -> pd.DataFrame:
        """Derive flat helper columns (raw_event_type, set_piece_type, is_key_pass,
        prop_* etc.) from the nested v1-format ``event_types`` list column.

        Shared by both the v1 (``load_event_data``) and v2 (``load_event_data_v2``)
        paths so the resulting ``self.events`` schema is identical regardless of
        the raw source format.
        """
        # ── Extract flat columns from event_types list ──
        priority = BeproDataPreprocessor._PRIMARY_EVENT_PRIORITY

        def _extract_primary(event_list: list[dict]) -> Optional[str]:
            names = {e.get("event_name") for e in event_list if e.get("event_name")}
            for name in priority:
                if name in names:
                    return name
            return None

        def _find_event_dict(event_list: list[dict], target: str) -> Optional[dict]:
            for e in event_list:
                if e.get("event_name") == target:
                    return e
            return None

        events["raw_event_type"] = events["event_types"].apply(_extract_primary)

        # Set piece type
        events["set_piece_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Set Pieces"),
                None,
            )
        )

        # Context boolean / categorical columns
        events["is_key_pass"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Key Passes" for e in el)
        )
        events["is_assist"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Assists" for e in el)
        )
        events["turnover_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Turnover"),
                None,
            )
        )
        events["has_recovery"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Recoveries" for e in el)
        )
        events["has_duel"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Duels" for e in el)
        )
        events["duel_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Duels"),
                None,
            )
        )
        events["received_type"] = events["event_types"].apply(
            lambda el: (
                "pass_received" if any(e.get("event_name") == "Passes Received" for e in el)
                else "cross_received" if any(e.get("event_name") == "Crosses Received" for e in el)
                else None
            )
        )

        # Helper flags for downstream SPADL conversion.
        # For multi-event rows, primary is picked by _PRIMARY_EVENT_PRIORITY but
        # secondary events (e.g. Tackles co-occurring with Passes) carry useful
        # context that would otherwise be lost.
        events["has_tackle"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Tackles" for e in el)
        )
        events["tackle_outcome"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Outcome")
                 for e in el if e.get("event_name") == "Tackles"),
                None,
            )
        )
        events["has_interception"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Interceptions" for e in el)
        )
        events["has_clearance"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Clearances" for e in el)
        )
        events["has_dls"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Defensive Line Supports" for e in el)
        ) # Defensive Line Support
        events["dls_outcome"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Outcome")
                 for e in el if e.get("event_name") == "Defensive Line Supports"),
                None,
            )
        )
        events["has_block"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Blocks" for e in el)
        )
        events["block_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Blocks"),
                None,
            )
        )
        events["has_offside"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Offsides" for e in el)
        )
        events["has_own_goal"] = events["event_types"].apply(
            lambda el: any(e.get("event_name") == "Own Goals" for e in el)
        )
        events["foul_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Fouls"),
                None,
            )
        )
        events["save_type"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Type")
                 for e in el if e.get("event_name") == "Saves"),
                None,
            )
        )
        events["aerial_control_outcome"] = events["event_types"].apply(
            lambda el: next(
                (e.get("property", {}).get("Outcome")
                 for e in el if e.get("event_name") == "Aerial Control"),
                None,
            )
        )

        # Primary event properties → flat columns
        def _extract_primary_props(row) -> dict:
            primary = row["raw_event_type"]
            if primary is None:
                return {}
            ed = _find_event_dict(row["event_types"], primary)
            if ed is None:
                return {}
            return ed.get("property", {}) or {}

        props_df = events.apply(_extract_primary_props, axis=1, result_type="expand")
        if not props_df.empty and len(props_df.columns) > 0:
            props_df.columns = [
                f"prop_{c.lower().replace(' ', '_')}" for c in props_df.columns
            ]
            events = pd.concat([events, props_df], axis=1)

        return events

    @staticmethod
    def load_tracking_data(tracking_path: List[str], raw_metadata: Dict, lineup: pd.DataFrame, target_fps: int) -> pd.DataFrame:
        player_lookup = lineup.set_index('player_id')
        home_tid = lineup[lineup['home_away'] == 'home']['team_id'].iloc[0]
        away_tid = lineup[lineup['home_away'] == 'away']['team_id'].iloc[0]
        
        first_half_tracking_data = []
        with open(tracking_path[0], 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                processed_line = line.strip()
                if not processed_line:
                    continue
                first_half_tracking_data.append(json.loads(processed_line))

        second_half_tracking_data = []
        with open(tracking_path[1], 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                processed_line = line.strip()
                if not processed_line:
                    continue
                second_half_tracking_data.append(json.loads(processed_line))

        all_object_rows = []
        for half_tracking_data in [first_half_tracking_data, second_half_tracking_data]:
            for frame_data in half_tracking_data:
                # Check ball state
                ball_state = frame_data.get('ball_state')
                if ball_state is None or ball_state == 'out':
                    new_ball_state = 'dead'
                    ball_owning_team_id = None
                else:
                    new_ball_state = 'alive'
                    ball_owning_team_id = home_tid if ball_state == 'home' else (away_tid if ball_state == 'away' else ball_state)
                    
                # Extract frame information
                frame_info = {
                    'game_id': raw_metadata.get('match_id'),
                    'period_id': frame_data.get('period_order') + 1,
                    'timestamp': frame_data.get('match_time'),
                    'frame_id': frame_data.get('frame_index'),
                    'ball_state': new_ball_state,
                    'ball_owning_team_id': ball_owning_team_id,
                }

                for object_type in ['players', 'balls']:
                    object_list = frame_data.get(object_type, [])
                    if object_list:
                        for object_data in object_list:
                            row_data = frame_info.copy()
                            row_data.update(object_data)
                            
                            if object_type == 'balls':
                                row_data.update({
                                    'id': 'ball',
                                    'team_id': 'ball',
                                    'position_name': 'ball',
                                    'object_id': 'ball',
                                })
                            else:
                                player_pID = str(object_data.get('player_id'))
                                row_data['id'] = player_pID
                                if player_pID in player_lookup.index:
                                    row_data['object_id'] = player_lookup.loc[player_pID, 'object_id']
                                    row_data['team_id'] = player_lookup.loc[player_pID, 'team_id']
                                    row_data['position_name'] = player_lookup.loc[player_pID, 'playing_position']
                                else:
                                    raise ValueError(f"Player ID {player_pID} not found in lineup data.\n{player_lookup}")
                            
                            # Remove unnecessary columns
                            row_data.pop('object', None)
                            row_data.pop('player_id', None)
                            all_object_rows.append(row_data)

        tracking_df = pd.DataFrame(all_object_rows)
        tracking_df['timestamp'] = pd.to_timedelta(tracking_df['timestamp'], unit='ms')

        # Rescale pitch coordinates
        tracking_df = BeproDataPreprocessor.rescale_pitch(tracking_df, raw_metadata)
        tracking_df = BeproDataPreprocessor.resample_tracking_dataframe(tracking_df, target_hz=target_fps)
        
        # Set z coordinate (bepro data doesn't have z information)
        tracking_df['z'] = 0.0
        # Drop rows with NaN coordinates for players (keep ball NaN rows)
        player_nan_mask = (tracking_df['id'] != 'ball') & tracking_df[['x', 'y']].isna().any(axis=1)
        total_tracking_df = tracking_df[~player_nan_mask].copy()
        
        # Sort and format final DataFrame
        total_tracking_df = total_tracking_df.sort_values(
            by=["period_id", "timestamp", "frame_id", "id"], kind="mergesort"
        ).reset_index(drop=True)

        # Define final column order
        final_cols_order = [
            'game_id', 'period_id', 'timestamp', 'frame_id', 'ball_state', 'ball_owning_team_id',
            'x', 'y', 'z',
            'id', 'object_id', 'team_id', 'position_name'
        ]
        total_tracking_df = total_tracking_df[final_cols_order]

        # determine ball owning team at each frame
        # Determine the ball-owning team from the provider tracking signal.
        g = total_tracking_df.groupby(['period_id', 'timestamp', 'frame_id'])
        ball_state = pd.DataFrame({
            "ball_state": g["ball_state"].apply(lambda s: s.value_counts().idxmax()),
            "ball_owning_team_id": g["ball_owning_team_id"].apply(
                lambda s: BeproDataPreprocessor.select_ball_owning_team_id(s, home_tid=home_tid, away_tid=away_tid)
            ),
        }).reset_index(drop=False) # index (period_id, timestamp, frame_id) to columns

        total_tracking_df = total_tracking_df.pivot_table(
            index= ['period_id', 'timestamp', 'frame_id'],
            columns='object_id',
            values=['x', 'y', 'z']#, 'vx', 'vy']
        )

        # value: x, y, vx, vy
        # player_code: H01, H02, A01, A02, ..., B00
        total_tracking_df.columns = [f'{player_code}_{value}' for value, player_code in total_tracking_df.columns]    
        total_tracking_df = total_tracking_df.reset_index(drop=False)

        total_tracking_df = total_tracking_df.merge(
            ball_state,
            how='left',
            on=['period_id', 'timestamp', 'frame_id']
        )
        total_tracking_df.loc[total_tracking_df['ball_x'].isna() | total_tracking_df['ball_y'].isna(), 'ball_state'] = 'dead'
        total_tracking_df.loc[total_tracking_df['ball_x'].isna() | total_tracking_df['ball_y'].isna(), 'ball_owning_team_id'] = pd.NA
    
        # Convert timestamps from milliseconds to seconds.
        total_tracking_df['timestamp'] = (
            total_tracking_df['timestamp'].dt.total_seconds()
            - ((total_tracking_df.period_id > 1) * 45 * 60)
            - ((total_tracking_df.period_id > 2) * 15 * 60)
            - ((total_tracking_df.period_id > 3) * 15 * 60)
        )

        total_tracking_df["ball_owning_team_id"] = BeproDataPreprocessor.normalize_ball_owning_team_id(
            total_tracking_df["ball_owning_team_id"],
            home_tid=home_tid,
            away_tid=away_tid,
        )
        total_tracking_df = BeproDataPreprocessor.order_tracking_columns(total_tracking_df, lineup)

        return total_tracking_df

    @staticmethod
    def select_ball_owning_team_id(ball_owning_team_id: pd.Series, home_tid: str, away_tid: str):
        valid_values = {str(home_tid), str(away_tid), "neutral"}
        cleaned = ball_owning_team_id.replace(
            {
                None: pd.NA,
                "None": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
            }
        ).dropna()
        cleaned = cleaned[cleaned.astype(str).isin(valid_values)]
        if cleaned.empty:
            return pd.NA
        return str(cleaned.value_counts().idxmax())

    @staticmethod
    def normalize_ball_owning_team_id(ball_owning_team_id: pd.Series, home_tid: str, away_tid: str) -> pd.Series:
        valid_values = {str(home_tid), str(away_tid), "neutral"}
        normalized = ball_owning_team_id.replace(
            {
                None: pd.NA,
                "None": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
            }
        ).astype("string")
        return normalized.where(normalized.isin(valid_values), pd.NA)

    @staticmethod
    def order_tracking_columns(tracking_df: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        fixed_cols = ["period_id", "timestamp", "frame_id", "ball_state", "ball_owning_team_id"]
        metric_order = ["x", "y", "z", "d", "s"]

        def object_metric_columns(object_id: str) -> List[str]:
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
    def rescale_pitch(tracking_df: pd.DataFrame, raw_metadata: Dict) -> pd.DataFrame:
        """Rescales pitch coordinates to standard dimensions.
        
        This function transforms pitch coordinates from the original coordinate system
        to a standardized pitch coordinate system. It handles both x and y coordinates
        and applies appropriate scaling factors based on the pitch metadata.
        
        Args:
            tracking_df: DataFrame containing tracking data with x, y coordinates.
            raw_metadata: Dictionary containing pitch metadata including ground_width and ground_height.
            
        Returns:
            DataFrame with rescaled x, y coordinates to standard pitch dimensions.
            
        Example:
            >>> rescaled_df = rescale_pitch(tracking_df, raw_metadata)
        """
        x_ori_min, x_ori_max = 0.0, raw_metadata['ground_width']
        y_ori_min, y_ori_max = 0.0, raw_metadata['ground_height']

        x_new_min, x_new_max = -PITCH_X / 2, PITCH_X / 2
        y_new_min, y_new_max = -PITCH_Y / 2, PITCH_Y / 2

        scale_x = (x_new_max - x_new_min) / (x_ori_max - x_ori_min)
        scale_y = (y_new_max - y_new_min) / (y_ori_max - y_ori_min)

        tracking_df['x'] = x_new_min + (tracking_df['x'] - x_ori_min) * scale_x
        tracking_df['y'] = y_new_min + (tracking_df['y'] - y_ori_min) * scale_y
        
        return tracking_df

    @staticmethod
    def resample_tracking_dataframe(tracking_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
        """Resamples tracking data to target frequency.
        
        This function resamples tracking data from its original frequency to a target
        frequency using interpolation techniques. It handles both forward and backward
        filling for different types of data columns and ensures proper time alignment.
        
        Args:
            tracking_df: DataFrame containing tracking data with timestamp index.
            target_hz: Target frequency in Hz for resampling.
            
        Returns:
            Resampled DataFrame with data at the target frequency.
            
        Example:
            >>> resampled_df = resample_tracking_dataframe(tracking_df, target_hz=25)
        """
        resample_freq_ms = int(1000 / target_hz)
        resample_freq_str = f'{resample_freq_ms}ms'
        
        period_list = []
        for period_id in tracking_df['period_id'].unique():
            period_df = tracking_df[tracking_df['period_id'] == period_id]

            min_timestamp = pd.Timedelta(0)
            max_timestamp = period_df['timestamp'].max()
            global_original_index = pd.to_timedelta(sorted(period_df['timestamp'].unique()))
            global_target_index = pd.timedelta_range(start=min_timestamp, end=max_timestamp, freq=resample_freq_str)

            grouped = period_df.groupby('id')
            resampled_list = []

            for agent_id, agent_group in grouped:
                group_df = agent_group.copy().set_index('timestamp')
                if group_df.index.has_duplicates:
                    group_df = group_df.loc[~group_df.index.duplicated(keep='first')]

                union_index = global_original_index.union(global_target_index)
                reindexed_group = group_df.reindex(union_index)

                # Interpolation
                interpolation_cols = ['x', 'y']
                reindexed_group[interpolation_cols] = reindexed_group[interpolation_cols].interpolate(method='pchip', limit_area='inside')
                
                # Forward fill other columns
                ffill_cols = [col for col in group_df.columns if col not in interpolation_cols and col != 'id']
                reindexed_group[ffill_cols] = reindexed_group[ffill_cols].ffill()
                final_group = reindexed_group.reindex(global_target_index)

                # Fill categorical data
                final_group['id'] = agent_id
                final_group = final_group.dropna(subset=['x', 'y'])
                resampled_list.append(final_group)

            period_list += resampled_list

        total_resampled_df = pd.concat(period_list).reset_index().rename(columns={'index': 'timestamp'})
        total_resampled_df['frame_id'] = (total_resampled_df['timestamp'].astype(np.int64) // (10**9 / target_hz)).astype(int)
        total_resampled_df = total_resampled_df.sort_values(['timestamp', 'period_id', 'frame_id', 'id'])

        return total_resampled_df

    @staticmethod
    def align_event_identifier(
        lineup: pd.DataFrame,
        events: pd.DataFrame,
        match_id: int,
    ) -> pd.DataFrame:
        events = events.copy()

        player_name_to_id = (
            lineup.dropna(subset=["player_name", "player_id"])
            .drop_duplicates("player_name")
            .set_index("player_name")["player_id"]
            .to_dict()
        )
        if "player_id" not in events.columns:
            events["player_id"] = events["player_name"].map(player_name_to_id)
        elif "player_name" in events.columns:
            events["player_id"] = events["player_id"].fillna(
                events["player_name"].map(player_name_to_id)
            )

        # Some v1 extracts contain a team_id column whose values are all null.
        # Treat those values as missing and recover them from the lineup instead
        # of checking only whether the column itself exists.
        player_id_to_team_id = (
            lineup.dropna(subset=["player_id", "team_id"])
            .assign(player_id=lambda frame: frame["player_id"].astype("string"))
            .drop_duplicates("player_id")
            .set_index("player_id")["team_id"]
            .astype("string")
            .to_dict()
        )
        mapped_team_ids = events["player_id"].astype("string").map(player_id_to_team_id)
        if "team_id" not in events.columns:
            events["team_id"] = mapped_team_ids
        else:
            events["team_id"] = events["team_id"].astype("string").fillna(mapped_team_ids)

        if "match_id" not in events.columns:
            events["match_id"] = match_id
        else:
            events["match_id"] = events["match_id"].fillna(match_id)

        if "event_id" not in events.columns:
            events["event_id"] = events.apply(
                lambda event: f"{event.match_id}_{event.period_id}_{event.name}",
                axis=1,
            )

        return events

    @staticmethod
    def align_event_orientations(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
            Rotate events so that the home team plays on the left side
        """
        events = events.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "home", "player_id"].tolist()
        away_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "away", "player_id"].tolist()

        for period_id in events["period_id"].unique():
            period_events = events[events["period_id"] == period_id].copy()
            home_gk_x = period_events.loc[period_events["player_id"].isin(home_gk_ids), "x"]
            away_gk_x = period_events.loc[period_events["player_id"].isin(away_gk_ids), "x"]

            if home_gk_x.mean() > away_gk_x.mean():
                events.loc[period_events.index, "x"] = (-period_events["x"]).round(2)
                events.loc[period_events.index, "to_x"] = (-period_events["to_x"]).round(2)
                events.loc[period_events.index, "y"] = (-period_events["y"]).round(2)
                events.loc[period_events.index, "to_y"] = (-period_events["to_y"]).round(2)

        return events

    @staticmethod
    def align_tracking_orientations(lineup: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
        """
            Rotate traces so that the home team plays on the left side
        """

        traces = traces.copy()

        gk_lineup = lineup.loc[lineup["playing_position"] == "GK"]
        home_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "home", "object_id"].tolist()
        away_gk_ids = gk_lineup.loc[gk_lineup["home_away"] == "away", "object_id"].tolist()

        for period_id in traces["period_id"].unique():
            period_traces = traces[traces["period_id"] == period_id].copy()
            home_gk_x = period_traces[[f"{p}_x" for p in home_gk_ids]].values
            away_gk_x = period_traces[[f"{p}_x" for p in away_gk_ids]].values
            
            x_cols = [col for col in traces.columns if col.endswith("_x")]
            y_cols = [col for col in traces.columns if col.endswith("_y")]

            if home_gk_x.mean() > away_gk_x.mean():
                traces.loc[period_traces.index, x_cols] = -traces.loc[period_traces.index, x_cols].values
                traces.loc[period_traces.index, y_cols] = -traces.loc[period_traces.index, y_cols].values

        return traces

    @staticmethod
    def find_object_ids(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
        events["object_id"] = events["player_id"].map(player_mapping)
        # bepro: not exist receiver_player_id
        events["receiver_id"] = np.nan # events["receiver_player_id"].map(player_mapping)

        return events

    @staticmethod
    def find_spadl_event_types(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events = convert_to_actions(events)

        return events

    def preprocess_event_data(self) -> pd.DataFrame:
        source_events = self.find_object_ids(self.lineup, self.events)
        events = self.find_spadl_event_types(source_events)
        events = events.merge(
            source_events[["event_id", "home_score", "away_score", "score"]],
            left_on="original_event_id",
            right_on="event_id",
            how="left",
        )
       
        # Bepro raw clocks exclude the halftime break, so reconstruct an
        # approximate UTC by adding a configurable halftime assumption to the
        # period-relative elapsed time.
        events["utc_timestamp"] = self._approx_utc_from_period_elapsed_seconds(
            self.raw_metadata,
            events["period_id"],
            events["time_seconds"],
            halftime_assumption_minutes=self.halftime_assumption_minutes,
        )
        
        selected_columns = [
        "period_id",
        "utc_timestamp",
        "team_id",
        "player_id",
        "object_id",
        "type_name",
        "start_x",
        "start_y",
        "success",
        "home_score",
        "away_score",
        "score",
        ]

        column_mapping = {
            "period_id": "period_id",
            "utc_timestamp": "utc_timestamp",
            "object_id": "object_id",
            "player_id": "player_id",
            "type_name": "spadl_type",
            "start_x": "start_x",
            "start_y": "start_y",
            "success": "success",
            "home_score": "home_score",
            "away_score": "away_score",
            "score": "score",
        }

        # input_events = events.loc[events["type_name"].notna(), column_mapping.keys()].copy().reset_index(drop=True)
        input_events = events.loc[events["type_name"].notna(), selected_columns].copy().reset_index(drop=True)
        input_events = input_events.rename(columns=column_mapping).astype({"success": bool})
        
        input_events["team_id"] = input_events["team_id"].astype("string")
        input_events["player_id"] = input_events["player_id"].astype("string")
        input_events["object_id"] = input_events["object_id"].astype("string")
        input_events["home_score"] = input_events["home_score"].astype("int64")
        input_events["away_score"] = input_events["away_score"].astype("int64")
        input_events["score"] = input_events["score"].astype("string")

        input_events = input_events[input_events["player_id"].notna()].reset_index(drop=True)
        # input_events = input_events[input_events["spadl_type"] != "dribble"].reset_index(drop=True)
        input_events["period_id"] = input_events["period_id"].map(CDF_PERIOD_MAP)

        return input_events
    
    # -------------------------------------------------------------------------
    # Raw events -> CDF
    # -------------------------------------------------------------------------

    # Bepro event_type derivation: maps raw_event_type to a normalized CDF-
    # friendly event_type.  For multi-event rows the derived type depends on
    # sub-elements (e.g. "Saves" → "keeper_save" / "keeper_punch" based on
    # the Save Type property).
    _RAW_TO_EVENT_TYPE: dict[str, str] = {
        "Passes": "pass",
        "Crosses": "cross",
        "Shots & Goals": "shot",
        "Tackles": "tackle",
        "Interceptions": "interception",
        "Clearances": "clearance",
        "Fouls": "foul",
        "Take-on": "take_on",
        "Step-in": "dribble",
        "Mistakes": "bad_touch",
        "Own Goals": "own_goal",
        "Blocks": "block",
        "Recoveries": "recovery",
        "Duels": "duel",
        "Passes Received": "pass_received",
        "Crosses Received": "cross_received",
        "Offsides": "offside",
        "Goals Conceded": "goals_conceded",
        "Set Piece Defence": "set_piece_defence",
        "Defensive Line Supports": "defensive_line_support",
    }

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
    def _derive_event_type(row: pd.Series) -> str:
        raw = row.get("raw_event_type")
        if raw is None or pd.isna(raw):
            return pd.NA

        base = BeproDataPreprocessor._RAW_TO_EVENT_TYPE.get(raw, raw)

        if raw == "Saves":
            save_type = row.get("prop_type")
            if pd.notna(save_type) and save_type == "Catches":
                return "keeper_save"
            if pd.notna(save_type) and save_type == "Parries":
                return "keeper_punch"
            return "keeper_save"

        if raw == "Aerial Control":
            return "keeper_claim"

        if raw == "Duels":
            duel_type = row.get("duel_type")
            if pd.notna(duel_type):
                if duel_type == "Aerial Duels":
                    return "aerial_duel"
                if duel_type == "Physical Duels":
                    return "ground_duel"
                if duel_type == "Loose Ball Duels":
                    return "loose_ball_duel"
            return "duel"

        return base

    # Map bepro raw set-piece labels to CDF-standard sub_type values
    # (per CDF paper: throw_in, free_kick, corner_kick, goal_kick, kick_off, penalty_kick).
    _SET_PIECE_TO_CDF_SUB_TYPE: dict[str, str] = {
        "Throw-Ins": "throw_in",
        "Freekicks": "free_kick",
        "Corners": "corner_kick",
        "Goal Kicks": "goal_kick",
        "Penalty Kicks": "penalty_kick",
        "Kick Off": "kick_off",
    }

    @staticmethod
    def _infer_cdf_sub_type(row: pd.Series) -> Optional[str]:
        set_piece = row.get("set_piece_type")
        if pd.notna(set_piece):
            mapped = BeproDataPreprocessor._SET_PIECE_TO_CDF_SUB_TYPE.get(set_piece)
            if mapped is not None:
                return mapped
            # Fallback: normalize unknown set-piece label
            return BeproDataPreprocessor._normalize_cdf_label(set_piece)

        # Duels are already encoded in `type` (aerial_duel / ground_duel /
        # loose_ball_duel) per the CDF paper's shot/pass/referee/misc grouping,
        # so no duel sub_type is emitted here.
        return pd.NA

    @staticmethod
    def _infer_cdf_outcome(row: pd.Series) -> Optional[str]:
        prop_outcome = row.get("prop_outcome")
        if pd.notna(prop_outcome):
            if prop_outcome in ("Succeeded", "Goals",
                                "Tackle Succeeded: Possession",
                                "Tackle Succeeded: No Possession"):
                return "successful"
            if prop_outcome in ("Failed", "Tackle Failed",
                                "Shots On Target", "Shots Off Target",
                                "Blocked Shots", "Keeper Rush-outs"):
                return "unsuccessful"
            if prop_outcome == "offside":
                return "offside"

        event_type = row.get("event_type")
        neutral_types = {"offside", "goals_conceded", "set_piece_defence",
                         "pass_received", "cross_received"}
        if pd.notna(event_type) and event_type in neutral_types:
            return "neutral"
        return pd.NA

    @staticmethod
    def _infer_cdf_outcome_detailed(row: pd.Series) -> Optional[str]:
        normalize = BeproDataPreprocessor._normalize_cdf_label
        prop_outcome = normalize(row.get("prop_outcome"))
        if prop_outcome is not pd.NA and pd.notna(prop_outcome):
            return prop_outcome
        return pd.NA

    @staticmethod
    def _infer_cdf_body_part(row: pd.Series) -> Optional[str]:
        event_type = row.get("event_type")
        raw = row.get("raw_event_type")
        duel_type = row.get("duel_type")
        foul_type = row.get("prop_type")

        # Aerial events → head
        if pd.notna(raw) and raw == "Aerial Control":
            return "head"
        if pd.notna(duel_type) and duel_type == "Aerial Duels":
            return "head"

        # Goalkeeper → other
        if pd.notna(event_type) and event_type in ("keeper_save", "keeper_punch", "keeper_claim"):
            return "other"

        # Fouls: handball → other
        if pd.notna(raw) and raw == "Fouls" and pd.notna(foul_type) and foul_type in ("Handball Foul", "Foul Throw"):
            return "other"

        # Default for most on-ball events
        if pd.notna(event_type) and event_type in ("pass", "cross", "shot", "tackle", "interception",
                          "clearance", "take_on", "dribble", "bad_touch",
                          "foul", "block", "recovery"):
            return "foot"

        return pd.NA

    @staticmethod
    def convert_raw_events_to_cdf(
        raw_events: pd.DataFrame,
        lineup: Optional[pd.DataFrame] = None,
        *,
        raw_metadata: Optional[dict] = None,
        align_orientations: bool = True,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.DataFrame:
        events = raw_events.copy()

        # ── Identifiers ──
        # align_event_identifier / add_score_columns are already applied in
        # __init__, but when called standalone they may be missing.
        if lineup is not None and "player_id" not in events.columns:
            events = BeproDataPreprocessor.align_event_identifier(
                lineup, events,
                match_id=events.get("match_id", pd.Series([pd.NA])).iloc[0],
            )

        if lineup is not None and align_orientations and "x" in events.columns:
            events = BeproDataPreprocessor.align_event_orientations(lineup, events)

        if lineup is not None and include_score and "home_score" not in events.columns:
            events = BeproDataPreprocessor.add_score_columns(events, lineup)

        # ── Derived event_type ──
        events["event_type"] = events.apply(BeproDataPreprocessor._derive_event_type, axis=1)

        # ── UTC timestamp ──
        if "utc_timestamp" not in events.columns and raw_metadata is not None:
            # Raw bepro event_time is a match clock that excludes halftime, not
            # a true UTC timestamp. Reconstruct an approximate UTC by converting
            # the raw match clock to period-relative elapsed time and injecting
            # a halftime assumption between the regular halves.
            events["utc_timestamp"] = BeproDataPreprocessor._approx_utc_from_match_clock_ms(
                raw_metadata,
                events["period_id"],
                events["event_time"],
                period_start_time_ms=events.get("period_start_time"),
                halftime_assumption_minutes=halftime_assumption_minutes,
            )

        # ── CDF columns ──
        events["period"] = events["period_id"].map(CDF_PERIOD_MAP).astype("string")
        events["type"] = events["event_type"].apply(
            BeproDataPreprocessor._normalize_cdf_label
        ).astype("string")
        events["sub_type"] = events.apply(
            BeproDataPreprocessor._infer_cdf_sub_type, axis=1
        ).astype("string")
        events["outcome"] = events.apply(
            BeproDataPreprocessor._infer_cdf_outcome, axis=1
        ).astype("string")
        events["outcome_detailed"] = events.apply(
            BeproDataPreprocessor._infer_cdf_outcome_detailed, axis=1
        ).astype("string")
        events["body_part"] = events.apply(
            BeproDataPreprocessor._infer_cdf_body_part, axis=1
        ).astype("string")

        # ── Coordinate aliases ──
        # Preserve raw coords before overwriting with CDF names
        events["source_x"] = events["x"]
        events["source_y"] = events["y"]
        events["source_to_x"] = events["to_x"]
        events["source_to_y"] = events["to_y"]
        # CDF uses plain x/y — same values, already center-origin
        events["x"] = events["source_x"]
        events["y"] = events["source_y"]
        events["x_end"] = events["to_x"]
        events["y_end"] = events["to_y"]

        # ── ID columns ──
        for col in ("match_id", "event_id", "team_id", "player_id"):
            if col in events.columns:
                events[col] = events[col].astype("string")

        events["receiver_id"] = pd.Series(pd.NA, index=events.index, dtype="string")
        events["receiver_time"] = pd.Series(pd.NA, index=events.index, dtype="string")
        events["related_event_ids"] = None

        if lineup is not None and not lineup.empty:
            player_mapping = lineup.set_index("player_id")["object_id"].to_dict()
            events["object_id"] = events["player_id"].map(player_mapping).astype("string")
            events["receiver_object_id"] = pd.Series(pd.NA, index=events.index, dtype="string")

        # ── Pass/Cross Received → fill receiver_id / receiver_time ──
        # Two sources of reception info:
        #  1) Rows where primary event IS pass_received/cross_received → remove after mapping
        #  2) Rows where received_type is set but primary is another action → keep row, use for mapping
        rcv_primary_mask = events["event_type"].isin(["pass_received", "cross_received"])
        rcv_secondary_mask = (
            events["received_type"].isin(["pass_received", "cross_received"])
            & ~rcv_primary_mask
        )
        pass_cross_mask = events["event_type"].isin(["pass", "cross"])

        if (rcv_primary_mask.any() or rcv_secondary_mask.any()) and pass_cross_mask.any():
            pc_indices = events.index[pass_cross_mask].tolist()

            def _fill_receiver(rcv_idx):
                rcv_row = events.loc[rcv_idx]
                rcv_period = rcv_row["period_id"]
                rcv_time = rcv_row["event_time"]
                candidates = [
                    i for i in pc_indices
                    if events.at[i, "period_id"] == rcv_period
                    and events.at[i, "event_time"] <= rcv_time
                ]
                if not candidates:
                    return
                match_idx = candidates[-1]
                events.at[match_idx, "receiver_id"] = str(rcv_row["player_id"])
                events.at[match_idx, "receiver_time"] = str(rcv_time)
                if lineup is not None and not lineup.empty:
                    rcv_obj = player_mapping.get(str(rcv_row["player_id"]))
                    if rcv_obj is not None:
                        events.at[match_idx, "receiver_object_id"] = str(rcv_obj)

            for rcv_idx in events.index[rcv_primary_mask | rcv_secondary_mask]:
                _fill_receiver(rcv_idx)

            # Remove only rows where reception is the primary event
            events = events[~rcv_primary_mask].reset_index(drop=True)

        # Tracking sync placeholders (bepro has no per-event tracking frame in raw data)
        events["is_synced"] = pd.Series(False, index=events.index, dtype="boolean")
        events["tracking_frame_id"] = pd.Series(pd.NA, index=events.index, dtype="Int64")
        events["tracking_frame_id_end"] = pd.Series(pd.NA, index=events.index, dtype="Int64")

        # ── Set Piece Defence → attach to corresponding shot ──
        spd_mask = events["raw_event_type"] == "Set Piece Defence"
        if spd_mask.any():
            spd_rows = events.loc[spd_mask, ["event_time", "player_id", "prop_outcome"]].copy()
            shot_mask = events["event_type"] == "shot"
            for _, spd_row in spd_rows.iterrows():
                # Find shot at the same event_time
                candidates = events.loc[
                    shot_mask & (events["event_time"] == spd_row["event_time"])
                ]
                if not candidates.empty:
                    shot_idx = candidates.index[0]
                    events.at[shot_idx, "set_piece_defence_player_id"] = spd_row["player_id"]
                    events.at[shot_idx, "set_piece_defence_outcome"] = spd_row["prop_outcome"]

        # ── Column ordering ──
        core_columns = [
            "match_id", "event_id", "utc_timestamp",
            "period", "type", "sub_type",
            "outcome", "outcome_detailed",
            "team_id", "player_id", "receiver_id", "receiver_time",
            "body_part",
            "x", "y", "x_end", "y_end",
            "is_synced", "tracking_frame_id", "tracking_frame_id_end",
            "related_event_ids",
        ]

        spadl_helper_columns = [
            "success", "object_id", "receiver_object_id",
            "home_score", "away_score", "score",
        ]
        # success: derive from outcome for convenience
        if "success" not in events.columns:
            events["success"] = (events["outcome"] == "successful")
        events["success"] = events["success"].astype("boolean")

        raw_helper_columns = [
            "period_id", "event_time",
            "raw_event_type", "event_type", "set_piece_type",
            "is_key_pass", "is_assist",
            "turnover_type", "has_recovery", "has_duel", "duel_type",
            "received_type",
            "has_tackle", "tackle_outcome",
            "has_interception", "has_clearance",
            "has_dls", "dls_outcome",
            "has_block", "block_type",
            "has_offside", "has_own_goal",
            "foul_type", "save_type", "aerial_control_outcome",
            "source_x", "source_y", "source_to_x", "source_to_y",
            "set_piece_defence_player_id", "set_piece_defence_outcome",
        ]

        prop_columns = [c for c in events.columns if c.startswith("prop_")]
        raw_helper_columns.extend(prop_columns)

        preserved_columns = [
            c for c in events.columns
            if c not in core_columns
            and c not in spadl_helper_columns
            and c not in raw_helper_columns
        ]

        ordered_columns = [c for c in core_columns if c in events.columns]
        if include_spadl_helpers:
            ordered_columns.extend([c for c in spadl_helper_columns if c in events.columns])
        if preserve_raw_columns:
            ordered_columns.extend([c for c in raw_helper_columns if c in events.columns])
            ordered_columns.extend(preserved_columns)

        return events.loc[:, ordered_columns].reset_index(drop=True)

    @staticmethod
    def to_cdf_events(
        raw_events: pd.DataFrame,
        lineup: Optional[pd.DataFrame] = None,
        *,
        raw_metadata: Optional[dict] = None,
        align_orientations: bool = True,
        include_score: bool = True,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.DataFrame:
        return BeproDataPreprocessor.convert_raw_events_to_cdf(
            raw_events,
            lineup=lineup,
            raw_metadata=raw_metadata,
            align_orientations=align_orientations,
            include_score=include_score,
            include_spadl_helpers=include_spadl_helpers,
            preserve_raw_columns=preserve_raw_columns,
            halftime_assumption_minutes=halftime_assumption_minutes,
        )

    def preprocess_cdf_events(
        self,
        raw_events: Optional[pd.DataFrame] = None,
        *,
        include_spadl_helpers: bool = True,
        preserve_raw_columns: bool = False,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.DataFrame:
        if raw_events is None:
            raw_events = self.events

        return self.to_cdf_events(
            raw_events,
            lineup=self.lineup,
            raw_metadata=self.raw_metadata,
            align_orientations=False,  # Already aligned in __init__
            include_score=True,
            include_spadl_helpers=include_spadl_helpers,
            preserve_raw_columns=preserve_raw_columns,
            halftime_assumption_minutes=halftime_assumption_minutes,
        )

    # -------------------------------------------------------------------------
    # CDF -> SPADL
    # -------------------------------------------------------------------------
    # Extended actions beyond the official 23-type SPADL vocabulary that we
    # still emit because bepro carries enough signal to recover them cleanly.
    _SPADL_EXTENDED_ACTIONS = {"ball_recovery", "dispossessed", "shot_block"}

    # Bepro set-piece labels in raw data use plural/mixed forms while CDF and
    # SPADL use consistent lowercase forms.  Sub-type values in CDF already
    # follow the paper (throw_in, goal_kick, corner_kick, free_kick, penalty_kick,
    # kick_off), so sub_type checks below use those standard names.

    @staticmethod
    def infer_spadl_types(
        events_cdf: pd.DataFrame,
        *,
        include_extended_actions: bool = True,
        split_secondary_recoveries: bool = False,
    ) -> pd.DataFrame:
        """Convert CDF event types to SPADL action types for bepro.

        Parameters
        ----------
        events_cdf
            CDF-format events table (output of ``preprocess_cdf_events``).
            Must carry bepro raw helper columns (``block_type``, ``has_recovery``,
            ``has_dls``, etc.), so call ``preprocess_cdf_events(preserve_raw_columns=True)``
            upstream.
        include_extended_actions
            If True, keep extended defensive actions (``ball_recovery``,
            ``dispossessed``, ``shot_block``) beyond the official 23-type
            SPADL vocabulary.  If False, these are dropped (spadl_type set to NA).
        split_secondary_recoveries
            - ``False`` (default): only rows whose CDF ``type == "recovery"``
              map to ``ball_recovery``.
            - ``True``: additionally emit synthetic ``ball_recovery`` rows for
              multi-event rows where ``has_recovery`` is True but the primary
              event was another action (e.g. pass preceded by a loose-ball
              recovery tagged on the same row).  The synthetic row is inserted
              just *before* the primary action.
        """

        events = events_cdf.copy().reset_index(drop=True)
        events["spadl_type"] = pd.NA

        def equal_notna(left, right) -> bool:
            return pd.notna(left) and pd.notna(right) and left == right

        def is_false(value) -> bool:
            return pd.notna(value) and bool(value) is False

        # ── Pass / Cross / Shot with set-piece specialization ──
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
        events.loc[shot_mask & (events["sub_type"] == "penalty_kick"), "spadl_type"] = "shot_penalty"

        # ── Take-on (bepro Take-on → take_on; Step-in → dribble) ──
        events.loc[events["type"] == "take_on", "spadl_type"] = "take_on"
        events.loc[events["type"] == "dribble", "spadl_type"] = "dribble"

        # ── Simple defensive / set actions ──
        events.loc[events["type"] == "interception", "spadl_type"] = "interception"
        events.loc[events["type"] == "tackle", "spadl_type"] = "tackle"
        events.loc[events["type"] == "clearance", "spadl_type"] = "clearance"
        events.loc[events["type"] == "foul", "spadl_type"] = "foul"

        # ── Goalkeeper actions ──
        events.loc[events["type"] == "keeper_save", "spadl_type"] = "keeper_save"
        events.loc[events["type"] == "keeper_punch", "spadl_type"] = "keeper_punch"
        events.loc[events["type"] == "keeper_claim", "spadl_type"] = "keeper_claim"

        # ── Mistakes / Own Goals → bad_touch ──
        events.loc[events["type"] == "bad_touch", "spadl_type"] = "bad_touch"
        events.loc[events["type"] == "own_goal", "spadl_type"] = "bad_touch"

        # ── Blocks (differential mapping by block_type) ──
        # Tir contré   → shot_block   (defender blocked a shot)
        # Cross Blocked → interception (defender blocked a cross — pass-like)
        # Untyped      → interception (typically a blocked pass)
        block_type_col = events.get("block_type", pd.Series(pd.NA, index=events.index))
        for i in events.index[events["type"] == "block"]:
            btype = block_type_col.iloc[i] if i < len(block_type_col) else pd.NA
            if pd.notna(btype):
                btype_str = str(btype).strip()
                if btype_str == "Tir contré":
                    events.at[i, "spadl_type"] = "shot_block"
                    continue
                if btype_str == "Cross Blocked":
                    events.at[i, "spadl_type"] = "interception"
                    continue
            # Untyped block — fall back to preceding action's type
            if i > 0:
                prev_type = events.at[i - 1, "type"]
                if prev_type == "shot":
                    events.at[i, "spadl_type"] = "shot_block"
                else:
                    # pass, cross, or anything else → treat as interception
                    events.at[i, "spadl_type"] = "interception"
            else:
                events.at[i, "spadl_type"] = "interception"

        # ── Defensive Line Support → tackle or interception ──
        # Follows the rule from bepro_actions._fix_defensive_line_support:
        # same player holds possession afterwards → interception, else → tackle.
        for i in events.index[events["type"] == "defensive_line_support"]:
            player_id = events.at[i, "player_id"]
            next_idx = i + 1 if (i + 1) in events.index else None
            if next_idx is not None and equal_notna(events.at[next_idx, "player_id"], player_id):
                events.at[i, "spadl_type"] = "interception"
            else:
                events.at[i, "spadl_type"] = "tackle"

        # ── Recoveries ──
        # Conservative (R-A) and aggressive (R-B) both map explicit
        # `type == recovery` → ball_recovery.  Aggressive additionally inserts
        # synthetic recoveries in `to_spadl_events` (after end-coordinate
        # copying) based on `has_recovery` on multi-event rows.
        events.loc[events["type"] == "recovery", "spadl_type"] = "ball_recovery"

        # ── Duels: excluded from SPADL (Option A) ──
        # aerial_duel / ground_duel / loose_ball_duel / duel are left with
        # spadl_type = NA, so they are dropped downstream.

        # ── Drop extended actions when requested ──
        if not include_extended_actions:
            events.loc[
                events["spadl_type"].isin(BeproDataPreprocessor._SPADL_EXTENDED_ACTIONS),
                "spadl_type",
            ] = pd.NA

        # ── Offside backpropagation ──
        # Bepro emits Offsides as its own row after the offside pass.  Propagate
        # the offside signal back to the preceding pass/cross in the same period
        # by the same team so SPADL `result_name` can be "offside".
        offside_mask = events["type"] == "offside"
        for i in events.index[offside_mask]:
            period = events.at[i, "period"]
            # Walk backwards within the same period
            for j in range(i - 1, -1, -1):
                if events.at[j, "period"] != period:
                    break
                jtype = events.at[j, "type"]
                if jtype in ("pass", "cross"):
                    # Mark the preceding pass/cross as offside
                    events.at[j, "outcome"] = "unsuccessful"
                    events.at[j, "outcome_detailed"] = "offside"
                    events.at[j, "success"] = False
                    break
                if jtype in ("shot", "take_on", "dribble", "clearance", "tackle",
                             "interception", "bad_touch"):
                    # Different on-ball action in between — stop
                    break

        # ── Success flag (rules mirrored from sportec) ──
        always_success = ["interception", "tackle", "dispossessed",
                          "ball_recovery", "shot_block"]
        always_failure = ["foul"]
        receiver_dependent = ["clearance", "bad_touch"]

        events.loc[events["spadl_type"].isin(always_success), "success"] = True
        events.loc[events["spadl_type"].isin(always_failure), "success"] = False

        dependent_events = events[events["spadl_type"].isin(receiver_dependent)]
        spadl_events = events[events["spadl_type"].notna()]
        if not spadl_events.empty:
            last_spadl_idx = spadl_events.index[-1]
            for i in dependent_events.index:
                if i == last_spadl_idx:
                    events.at[i, "success"] = False
                else:
                    period = events.at[i, "period"]
                    team_id = events.at[i, "team_id"]
                    remaining = spadl_events.loc[i + 1:]
                    if remaining.empty:
                        events.at[i, "success"] = False
                    else:
                        next_event = remaining.iloc[0]
                        events.at[i, "success"] = bool(
                            next_event["period"] == period
                            and pd.notna(next_event["team_id"])
                            and pd.notna(team_id)
                            and next_event["team_id"] == team_id
                        )

        return events

    @staticmethod
    def _insert_synthetic_recoveries(events: pd.DataFrame) -> pd.DataFrame:
        """Insert synthetic ball_recovery rows for multi-event rows where
        has_recovery is True but the primary event was another action.

        Used when split_secondary_recoveries=True.  Each synthetic row inherits
        the primary event's context (team, player, coordinates, period) and is
        ordered immediately *before* the primary via _spadl_order - 0.1.
        """
        if "has_recovery" not in events.columns:
            return events

        secondary_mask = (
            events["has_recovery"].fillna(False).astype(bool)
            & (events["type"] != "recovery")
            & events["spadl_type"].notna()
        )
        if not secondary_mask.any():
            return events

        events = events.copy()
        if "_spadl_order" not in events.columns:
            events["_spadl_order"] = pd.Series(range(len(events)), index=events.index, dtype="float64")

        synthetic_rows = []
        for idx in events.index[secondary_mask]:
            extra = events.loc[idx].copy()
            extra["spadl_type"] = "ball_recovery"
            extra["success"] = True
            extra["sub_type"] = pd.NA
            extra["outcome"] = "successful"
            extra["outcome_detailed"] = pd.NA
            extra["receiver_id"] = pd.NA
            if "receiver_object_id" in events.columns:
                extra["receiver_object_id"] = pd.NA
            extra["body_part"] = "foot"
            # End coordinates collapse to start for a ball recovery
            extra["x_end"] = events.at[idx, "x"]
            extra["y_end"] = events.at[idx, "y"]
            if "end_x" in events.columns:
                extra["end_x"] = events.at[idx, "x"]
                extra["end_y"] = events.at[idx, "y"]
            extra["_spadl_order"] = float(events.at[idx, "_spadl_order"]) - 0.1
            synthetic_rows.append(extra)

        if not synthetic_rows:
            return events

        extra_df = pd.DataFrame(synthetic_rows).reindex(columns=events.columns)
        events = pd.concat([events, extra_df], ignore_index=True, sort=False)
        events = events.sort_values("_spadl_order", kind="stable").reset_index(drop=True)
        return events

    @staticmethod
    def _infer_spadl_bodypart_name(events: pd.DataFrame) -> pd.Series:
        # Bepro already encodes body_part as {foot, head, other} via
        # _infer_cdf_body_part. Pass through and ensure the value is one of the
        # SPADL-compatible names.
        valid = {"foot", "head", "other", "foot_left", "foot_right"}
        body_part = events["body_part"].astype("string")
        return body_part.where(body_part.isin(valid), pd.NA)

    @staticmethod
    def _infer_spadl_result_name(events: pd.DataFrame) -> pd.Series:
        result_name = pd.Series(pd.NA, index=events.index, dtype="string")
        result_name.loc[events["success"] == True] = "success"
        result_name.loc[events["success"] == False] = "fail"

        # Offside (backpropagated in infer_spadl_types)
        result_name.loc[events["outcome_detailed"] == "offside"] = "offside"
        # Own goal → owngoal
        result_name.loc[events["type"] == "own_goal"] = "owngoal"
        # Card handling (bepro Fouls carry Yellow Cards / Red Cards as foul_type)
        if "foul_type" in events.columns:
            yellow_mask = (
                (events["type"] == "foul")
                & (events["foul_type"].astype("string") == "Yellow Cards")
            )
            red_mask = (
                (events["type"] == "foul")
                & (events["foul_type"].astype("string") == "Red Cards")
            )
            result_name.loc[yellow_mask] = "yellow_card"
            result_name.loc[red_mask] = "red_card"
        return result_name

    @staticmethod
    def _copy_spadl_end_coordinates(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        events["end_x"] = pd.to_numeric(events["x_end"], errors="coerce")
        events["end_y"] = pd.to_numeric(events["y_end"], errors="coerce")
        return events

    @staticmethod
    def _enrich_spadl_end_coordinates(events: pd.DataFrame) -> pd.DataFrame:
        """Fill missing pass/cross end coordinates using the receiver's next
        tracked position (mirrors the sportec heuristic)."""
        events = events.copy()
        events["end_x"] = pd.to_numeric(events["x_end"], errors="coerce")
        events["end_y"] = pd.to_numeric(events["y_end"], errors="coerce")

        pass_like = {
            "pass", "cross",
            "throw_in",
            "freekick_short", "freekick_crossed",
            "corner_short", "corner_crossed",
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
    def to_spadl_events(
        events_cdf: pd.DataFrame,
        *,
        infer_end_coordinates: bool = False,
        include_cdf_columns: bool = False,
        include_extended_actions: bool = True,
        split_secondary_recoveries: bool = False,
        lineup: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        events = BeproDataPreprocessor.infer_spadl_types(
            events_cdf,
            include_extended_actions=include_extended_actions,
            split_secondary_recoveries=split_secondary_recoveries,
        )
        events["original_event_id"] = events["event_id"].astype("string") if "event_id" in events.columns else pd.NA
        events["_spadl_order"] = pd.Series(range(len(events)), index=events.index, dtype="float64")

        if split_secondary_recoveries:
            events = BeproDataPreprocessor._insert_synthetic_recoveries(events)

        if infer_end_coordinates:
            events = BeproDataPreprocessor._enrich_spadl_end_coordinates(events)
        else:
            events = BeproDataPreprocessor._copy_spadl_end_coordinates(events)
        events["bodypart_name"] = BeproDataPreprocessor._infer_spadl_bodypart_name(events)

        # take_on has no receiver in SPADL semantics
        take_on_mask = events["spadl_type"] == "take_on"
        if take_on_mask.any():
            events.loc[take_on_mask, "receiver_id"] = pd.NA
            if "receiver_object_id" in events.columns:
                events.loc[take_on_mask, "receiver_object_id"] = pd.NA

        if "success" not in events.columns:
            events["success"] = (events["outcome"] == "successful").astype("boolean")
        # Any remaining NA success defaults to False so result_name is always set
        # for rows that will be emitted (spadl_type not NA).
        spadl_mask = events["spadl_type"].notna()
        events.loc[spadl_mask & events["success"].isna(), "success"] = False
        events["result_name"] = BeproDataPreprocessor._infer_spadl_result_name(events)

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
            column_mapping.update({
                "type": "cdf_type",
                "sub_type": "cdf_sub_type",
                "outcome_detailed": "cdf_outcome_detailed",
            })

        available_columns = [c for c in selected_columns if c in events.columns]
        input_events = events.loc[events["spadl_type"].notna(), available_columns].copy().reset_index(drop=True)
        if "_spadl_order" in input_events.columns:
            input_events = input_events.sort_values("_spadl_order", kind="stable").reset_index(drop=True)
            input_events = input_events.drop(columns="_spadl_order")
        input_events = input_events.rename(columns=column_mapping)

        if "success" in input_events.columns:
            # Any remaining NA success (e.g. from CDF rows where outcome was NA)
            # defaults to False so that SPADL success is a strict bool column.
            input_events["success"] = input_events["success"].fillna(False).astype(bool)

        for column in (
            "match_id", "original_event_id",
            "team_id", "player_id",
            "object_id", "receiver_id", "receiver_object_id",
            "score", "period_id",
            "bodypart_name", "result_name",
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
        split_secondary_recoveries: bool = False,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.DataFrame:
        """Produce a SPADL-format action table from the bepro CDF events.

        Parameters
        ----------
        events_cdf
            Optional pre-computed CDF events table.  When omitted, the method
            calls ``preprocess_cdf_events(preserve_raw_columns=True)`` so that
            bepro-specific helper columns (block_type, has_recovery, ...) are
            available for SPADL inference.
        split_secondary_recoveries
            If True, emit an extra synthetic ``ball_recovery`` row for each
            multi-event row where ``has_recovery`` is True but the primary
            event was another action.  See ``infer_spadl_types`` for details.
        """
        if events_cdf is None:
            events_cdf = self.preprocess_cdf_events(
                preserve_raw_columns=True,
                halftime_assumption_minutes=halftime_assumption_minutes,
            )
        else:
            required = ("block_type", "has_recovery", "has_dls", "dls_outcome",
                        "foul_type")
            missing = [c for c in required if c not in events_cdf.columns]
            if missing:
                enriched = self.preprocess_cdf_events(
                    preserve_raw_columns=True,
                    halftime_assumption_minutes=halftime_assumption_minutes,
                )
                helper = ["event_id", *missing]
                events_cdf = events_cdf.merge(
                    enriched.loc[:, [c for c in helper if c in enriched.columns]],
                    on="event_id",
                    how="left",
                )

        return self.to_spadl_events(
            events_cdf,
            infer_end_coordinates=infer_end_coordinates,
            include_cdf_columns=include_cdf_columns,
            include_extended_actions=include_extended_actions,
            split_secondary_recoveries=split_secondary_recoveries,
            lineup=self.lineup,
        )

    def synchronize_spadl_events(
        self,
        events_spadl: Optional[pd.DataFrame] = None,
        tracking: Optional[pd.DataFrame] = None,
        *,
        apply_kinematic_correction: bool = False,
        split_secondary_recoveries: bool = False,
        fps: Optional[int] = None,
        args: Optional[dict] = None,
        filter_dead_ball_events: bool = True,
        dead_ball_window_seconds: float = 1.0,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> pd.DataFrame:
        from ..synchronize import synchronize_spadl_with_tracking

        if events_spadl is None:
            events_spadl = self.preprocess_spadl_events(
                split_secondary_recoveries=split_secondary_recoveries,
                halftime_assumption_minutes=halftime_assumption_minutes,
            )

        if tracking is None:
            _, tracking = self.preprocess_tracking_data(
                apply_kinematic_correction=apply_kinematic_correction,
                halftime_assumption_minutes=halftime_assumption_minutes,
            )

        if fps is None:
            fps = int(self.target_fps)

        event_filter = None
        if filter_dead_ball_events:
            window = float(dead_ball_window_seconds)

            def event_filter(events: pd.DataFrame, tracking_df: pd.DataFrame) -> pd.DataFrame:
                return self._filter_events_with_alive_tracking(
                    events, tracking_df, window_seconds=window,
                )

        return synchronize_spadl_with_tracking(
            events_spadl,
            tracking,
            lineup=self.lineup,
            fps=fps,
            args=args,
            event_filter=event_filter,
        )

    @staticmethod
    def _filter_events_with_alive_tracking(
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        *,
        window_seconds: float = 1.0,
    ) -> pd.DataFrame:
        if events.empty or tracking.empty:
            return events

        window = pd.Timedelta(seconds=window_seconds)
        ball_status = tracking["ball_status"].astype(bool)
        alive_tracking = tracking.loc[ball_status, ["period", "utc_timestamp"]]

        keep_indices = []
        for period_id, group in events.groupby("period_id"):
            period_alive = alive_tracking.loc[alive_tracking["period"] == period_id, "utc_timestamp"]
            if period_alive.empty:
                continue

            alive_ts = np.sort(pd.to_datetime(period_alive.values).astype("datetime64[ns]"))
            event_ts = pd.to_datetime(group["utc_timestamp"].values).astype("datetime64[ns]")

            left = np.searchsorted(alive_ts, event_ts - window, side="left")
            right = np.searchsorted(alive_ts, event_ts + window, side="right")
            has_alive = right > left

            keep_indices.extend(group.index[has_alive].tolist())

        return events.loc[keep_indices]


    @staticmethod
    def has_goal_tag(event_tags) -> bool:
        # Treat only successful shot events as goals.
        return any(
            item.get("event_name") == "Shots & Goals"
            and item.get("property", {}).get("Outcome") == "Goals"
            for item in (event_tags or [])
            if isinstance(item, dict)
        )

    @staticmethod
    def has_own_goal_tag(event_tags) -> bool:
        # Own goals are recorded as a separate event tag.
        return any(
            item.get("event_name") == "Own Goals"
            for item in (event_tags or [])
            if isinstance(item, dict)
        )

    @staticmethod
    def build_side_lookup(lineup: pd.DataFrame) -> dict:
        # Fallback maps for resolving home/away when event-side identifiers are incomplete.
        lineup = lineup.copy()

        lineup["team_id"] = lineup["team_id"].astype("string")
        lineup["player_id"] = lineup["player_id"].astype("string")
        lineup["team_name"] = lineup["team_name"].astype("string")
        lineup["player_name"] = lineup["player_name"].astype("string")

        team_id_to_side = (
            lineup[["team_id", "home_away"]]
            .dropna()
            .drop_duplicates()
            .set_index("team_id")["home_away"]
            .to_dict()
        )

        player_id_to_side = (
            lineup[["player_id", "home_away"]]
            .dropna()
            .drop_duplicates()
            .set_index("player_id")["home_away"]
            .to_dict()
        )

        team_name_to_side = (
            lineup[["team_name", "home_away"]]
            .dropna()
            .drop_duplicates()
            .set_index("team_name")["home_away"]
            .to_dict()
        )

        player_name_df = lineup[["player_name", "home_away"]].dropna().drop_duplicates()
        player_name_counts = player_name_df.groupby("player_name")["home_away"].nunique()
        unique_player_names = player_name_counts[player_name_counts == 1].index
        player_name_to_side = (
            player_name_df[player_name_df["player_name"].isin(unique_player_names)]
            .drop_duplicates("player_name")
            .set_index("player_name")["home_away"]
            .to_dict()
        )

        return {
            "team_id": team_id_to_side,
            "player_id": player_id_to_side,
            "team_name": team_name_to_side,
            "player_name": player_name_to_side,
        }

    @staticmethod
    def resolve_event_side(row: pd.Series, side_lookup: dict) -> str | None:
        team_id = row.get("team_id")
        if pd.notna(team_id):
            side = side_lookup["team_id"].get(str(team_id))
            if side is not None:
                return side

        player_id = row.get("player_id")
        if pd.notna(player_id):
            side = side_lookup["player_id"].get(str(player_id))
            if side is not None:
                return side

        team_name = row.get("team_name")
        if pd.notna(team_name):
            side = side_lookup["team_name"].get(str(team_name))
            if side is not None:
                return side

        player_name = row.get("player_name")
        if pd.notna(player_name):
            side = side_lookup["player_name"].get(str(player_name))
            if side is not None:
                return side

        return None

    @staticmethod
    def add_score_columns(events: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        side_lookup = BeproDataPreprocessor.build_side_lookup(lineup)

        home_score = 0
        away_score = 0
        home_scores = []
        away_scores = []

        pending_home_delta = 0
        pending_away_delta = 0

        for _, row in events.iterrows():
            home_score += pending_home_delta
            away_score += pending_away_delta
            pending_home_delta = 0
            pending_away_delta = 0

            home_scores.append(home_score)
            away_scores.append(away_score)

            side = BeproDataPreprocessor.resolve_event_side(row, side_lookup)

            if BeproDataPreprocessor.has_goal_tag(row.get("event_types")):
                if side == "home":
                    pending_home_delta = 1
                elif side == "away":
                    pending_away_delta = 1

            elif BeproDataPreprocessor.has_own_goal_tag(row.get("event_types")):
                if side == "home":
                    pending_away_delta = 1
                elif side == "away":
                    pending_home_delta = 1

        events["home_score"] = home_scores
        events["away_score"] = away_scores
        events["score"] = [f"{h}-{a}" for h, a in zip(home_scores, away_scores)]

        return events

    
    def preprocess_tracking_data(
        self,
        apply_kinematic_correction: bool = False,
        *,
        halftime_assumption_minutes: float = DEFAULT_HALFTIME_ASSUMPTION_MINUTES,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tracking = self.tracking.copy()
        tracking = tracking.drop(columns=["frame_id"])

        if "frame_id" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            tracking = BaseEventTrackingPreprocessor.calculate_tracking_datetimes(events=None, tracking=tracking, fps=self.target_fps)
            # After resampling, bepro tracking timestamps are period-relative
            # elapsed times (each half starts again at 0). Reconstruct an
            # approximate UTC directly from the elapsed time within period.
            elapsed_seconds = tracking["timestamp"]
            if pd.api.types.is_timedelta64_dtype(elapsed_seconds):
                elapsed_seconds = elapsed_seconds.dt.total_seconds()
            tracking["utc_timestamp"] = self._approx_utc_from_period_elapsed_seconds(
                self.raw_metadata,
                tracking["period_id"],
                elapsed_seconds,
                halftime_assumption_minutes=halftime_assumption_minutes,
            )

        return self._finalize_tracking_output(
            tracking,
            fps=self.target_fps,
            apply_kinematic_correction=apply_kinematic_correction,
        )
