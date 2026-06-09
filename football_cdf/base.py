import fnmatch
from abc import ABC, abstractmethod
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .constants import CDF_PERIOD_MAP, KINEMATIC_CORRECTION_DEFAULTS, SMOOTHING_DEFAULTS
from .utils import timestamp_to_seconds


class BaseEventTrackingPreprocessor(ABC):

    MATCH_METADATA_MINIMAL_COLUMNS = (
        "competition_id",
        "competition_name",
        "season_id",
        "season_name",
        "match_id",
        "kickoff_time",
        "play_direction",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "stadium_id",
        "stadium_name",
        "pitch_length",
        "pitch_width",
        "final_home_score",
        "final_away_score",
        "final_score",
        "vendor_name",
        "vendor_version",
        "cdf_version",
    )

    LINEUP_MINIMAL_COLUMNS = (
        "team_id",
        "team_name",
        "home_away",
        "player_id",
        "uniform_number",
        "object_id",
        "player_name",
        "playing_position",
        "starting",
    )

    def __init__(self):
        self.lineup: pd.DataFrame
        self.events: pd.DataFrame
        self.tracking: pd.DataFrame
        self.fps: float

    @staticmethod
    def _cast_ids_to_object(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        id_columns = [col for col in df.columns if col.endswith("_id")]
        for column in id_columns:
            df[column] = df[column].astype(object)
        return df

    @staticmethod
    def _cast_numeric_output(
        df: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> pd.DataFrame:
        df = df.copy()
        for column in columns:
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.isna().any():
                df[column] = numeric.astype("Int64")
            else:
                df[column] = numeric.astype("int64")
        return df

    @staticmethod
    def _normalize_match_metadata_output(df: pd.DataFrame) -> pd.DataFrame:
        df = BaseEventTrackingPreprocessor._cast_ids_to_object(df)
        df = BaseEventTrackingPreprocessor._cast_numeric_output(
            df,
            columns=("pitch_length", "pitch_width"),
        )
        return df

    @staticmethod
    def _normalize_lineup_output(df: pd.DataFrame) -> pd.DataFrame:
        df = BaseEventTrackingPreprocessor._cast_ids_to_object(df)
        df = BaseEventTrackingPreprocessor._cast_numeric_output(
            df,
            columns=("uniform_number",),
        )
        if "home_away" in df.columns:
            home_away = df["home_away"].astype("string")
            df["home_away"] = pd.Categorical(home_away, categories=["home", "away"])
        return df

    def get_match_metadata(self, mode: str = "full") -> pd.DataFrame:
        match_metadata = pd.DataFrame([self.match_metadata])
        if mode == "full":
            return self._normalize_match_metadata_output(match_metadata).reset_index(drop=True)
        if mode == "minimal":
            columns = [col for col in self.MATCH_METADATA_MINIMAL_COLUMNS if col in match_metadata.columns]
            output = match_metadata.loc[:, columns].copy()
            return self._normalize_match_metadata_output(output).reset_index(drop=True)
        raise ValueError(f"Unsupported match metadata mode: {mode}")

    def get_lineup(self, mode: str = "full") -> pd.DataFrame:
        if mode == "full":
            return self._normalize_lineup_output(self.lineup).reset_index(drop=True)
        if mode == "minimal":
            columns = [col for col in self.LINEUP_MINIMAL_COLUMNS if col in self.lineup.columns]
            output = self.lineup.loc[:, columns].copy()
            return self._normalize_lineup_output(output).reset_index(drop=True)
        raise ValueError(f"Unsupported lineup mode: {mode}")

    @staticmethod
    def calculate_event_seconds(events: pd.DataFrame) -> pd.DataFrame:
        assert "utc_timestamp" in events.columns  # in datetime
        events = events.copy()
        events["timestamp"] = 0.0

        for i in events["period_id"].unique():
            period_events: pd.DataFrame = events[events["period_id"] == i].copy()
            start_dt = period_events["utc_timestamp"].iloc[0]
            period_tds = period_events["utc_timestamp"] - start_dt
            events.loc[period_events.index, "timestamp"] = period_tds.dt.total_seconds()
            # events.loc[period_events.index, "utc_timestamp"] -= timedelta(microseconds=start_dt.microsecond)

        return events

    @staticmethod
    def calculate_tracking_datetimes(events: pd.DataFrame, tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
        assert "timestamp" in tracking.columns  # in seconds
        tracking = tracking.copy()

        if "frame_id" not in tracking.columns:
            tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
            max_frame_p1 = tracking.loc[tracking["period_id"] == 1, "frame_id"].max()
            tracking.loc[tracking["period_id"] == 2, "frame_id"] += max_frame_p1 + 1

        def utc_timestamp(t: float, offset: np.datetime64) -> np.datetime64:
            return offset + timedelta(seconds=t)

        if events is not None:
            tracking["utc_timestamp"] = pd.NaT
            for i in events["period_id"].unique():
                offset = events[events["period_id"] == i]["utc_timestamp"].iloc[0]
                period_tracking = tracking[tracking["period_id"] == i]
                period_ts = period_tracking["timestamp"].apply(utc_timestamp, args=(offset,))
                tracking.loc[period_ts.index, "utc_timestamp"] = period_ts

        return tracking

    @staticmethod
    @abstractmethod
    def load_raw_metadata(meta_path: str):
        """Load provider-native raw metadata."""
        pass

    @staticmethod
    @abstractmethod
    def extract_match_metadata(raw_metadata) -> dict:
        """Build canonical match metadata from provider-native raw metadata."""
        pass

    @staticmethod
    @abstractmethod
    def load_tracking_data(*args, **kwargs) -> pd.DataFrame:
        """Load provider tracking into a wide canonical DataFrame."""
        pass

    @abstractmethod
    def preprocess_event_data(self) -> pd.DataFrame:
        pass

    @staticmethod
    def _apply_savgol(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
        if len(values) == 0:
            return values
        fitted_window = min(window_length, len(values))
        if fitted_window % 2 == 0:
            fitted_window -= 1
        if fitted_window < polyorder + 1 or fitted_window <= 0:
            return values
        return savgol_filter(values, window_length=fitted_window, polyorder=polyorder)

    @staticmethod
    def _smooth_masked(series: pd.Series, mask: pd.Series, window_length: int, polyorder: int) -> pd.Series:
        smoothed = series.replace([np.inf, -np.inf], np.nan).mask(mask)
        smoothed = smoothed.interpolate(limit_direction="both")
        values = smoothed.fillna(0.0).to_numpy(dtype=float)
        values = BaseEventTrackingPreprocessor._apply_savgol(values, window_length=window_length, polyorder=polyorder)
        return pd.Series(values, index=series.index, dtype=float)

    def _calculate_object_kinematics(
        self,
        period_tracking: pd.DataFrame,
        fps: float,
        *,
        is_ball: bool,
        apply_kinematic_correction: bool = False,
    ) -> pd.DataFrame:
        period_tracking = period_tracking.copy()

        if apply_kinematic_correction:
            cfg = KINEMATIC_CORRECTION_DEFAULTS["ball" if is_ball else "player"]
            dt = period_tracking["timestamp"].diff().replace(0, np.nan)

            vx_raw = period_tracking["x"].diff() / dt
            vy_raw = period_tracking["y"].diff() / dt
            speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)

            speed_outlier = speed_raw > cfg["max_speed"]
            period_tracking["vx"] = self._smooth_masked(vx_raw, speed_outlier, cfg["window_length"], cfg["polyorder"])
            period_tracking["vy"] = self._smooth_masked(vy_raw, speed_outlier, cfg["window_length"], cfg["polyorder"])
            period_tracking["speed"] = np.sqrt(period_tracking["vx"]**2 + period_tracking["vy"]**2).clip(
                upper=cfg["max_speed"]
            )

            ax_raw = period_tracking["vx"].diff() / dt
            ay_raw = period_tracking["vy"].diff() / dt
            accel_v_raw = np.sqrt(ax_raw**2 + ay_raw**2)
            accel_outlier = accel_v_raw > cfg["max_acceleration"]

            period_tracking["ax"] = self._smooth_masked(ax_raw, accel_outlier, cfg["window_length"], cfg["polyorder"])
            period_tracking["ay"] = self._smooth_masked(ay_raw, accel_outlier, cfg["window_length"], cfg["polyorder"])
            period_tracking["accel_v"] = np.sqrt(period_tracking["ax"]**2 + period_tracking["ay"]**2).clip(
                upper=cfg["max_acceleration"]
            )

            accel_s_raw = period_tracking["speed"].diff() / dt
            accel_s_outlier = accel_s_raw.abs() > cfg["max_acceleration"]
            period_tracking["accel_s"] = self._smooth_masked(
                accel_s_raw, accel_s_outlier, cfg["window_length"], cfg["polyorder"]
            ).clip(lower=-cfg["max_acceleration"], upper=cfg["max_acceleration"])
        else:
            v_cfg = SMOOTHING_DEFAULTS["velocity"]
            a_cfg = SMOOTHING_DEFAULTS["acceleration"]

            vx_raw = pd.Series(np.nan, index=period_tracking.index, dtype=float)
            vy_raw = pd.Series(np.nan, index=period_tracking.index, dtype=float)
            vx_raw.iloc[1:] = np.diff(period_tracking["x"].to_numpy(dtype=float)) * fps
            vy_raw.iloc[1:] = np.diff(period_tracking["y"].to_numpy(dtype=float)) * fps

            period_tracking["vx"] = self._smooth_masked(
                vx_raw, pd.Series(False, index=period_tracking.index),
                v_cfg["window_length"], v_cfg["polyorder"],
            )
            period_tracking["vy"] = self._smooth_masked(
                vy_raw, pd.Series(False, index=period_tracking.index),
                v_cfg["window_length"], v_cfg["polyorder"],
            )
            period_tracking["speed"] = np.sqrt(period_tracking["vx"]**2 + period_tracking["vy"]**2)

            ax_raw = pd.Series(np.nan, index=period_tracking.index, dtype=float)
            ay_raw = pd.Series(np.nan, index=period_tracking.index, dtype=float)
            accel_s_raw = pd.Series(np.nan, index=period_tracking.index, dtype=float)
            ax_raw.iloc[1:] = np.diff(period_tracking["vx"].to_numpy(dtype=float)) * fps
            ay_raw.iloc[1:] = np.diff(period_tracking["vy"].to_numpy(dtype=float)) * fps
            accel_s_raw.iloc[1:] = np.diff(period_tracking["speed"].to_numpy(dtype=float)) * fps

            period_tracking["ax"] = self._smooth_masked(
                ax_raw, pd.Series(False, index=period_tracking.index),
                a_cfg["window_length"], a_cfg["polyorder"],
            )
            period_tracking["ay"] = self._smooth_masked(
                ay_raw, pd.Series(False, index=period_tracking.index),
                a_cfg["window_length"], a_cfg["polyorder"],
            )
            period_tracking["accel_s"] = self._smooth_masked(
                accel_s_raw, pd.Series(False, index=period_tracking.index),
                a_cfg["window_length"], a_cfg["polyorder"],
            )
            period_tracking["accel_v"] = np.sqrt(period_tracking["ax"]**2 + period_tracking["ay"]**2)

        kinematic_cols = ["vx", "vy", "ax", "ay", "speed", "accel_s", "accel_v"]
        period_tracking[kinematic_cols] = period_tracking[kinematic_cols].bfill().ffill()
        return period_tracking

    def _finalize_tracking_output(
        self,
        tracking: pd.DataFrame,
        fps: float,
        *,
        apply_kinematic_correction: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        home_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "home_*_x")]
        away_players = [c[:-2] for c in tracking.columns if fnmatch.fnmatch(c, "away_*_x")]
        objects = home_players + away_players + ["ball"]
        tracking_list = []

        object_to_player = self.lineup.set_index("object_id")["player_id"].to_dict()

        for p in objects:
            object_tracking = tracking[
                ["frame_id", "period_id", "timestamp", "utc_timestamp", "ball_state", "ball_owning_team_id"]
            ].copy()

            if p == "ball":
                object_tracking["player_id"] = pd.NA
                object_tracking["object_id"] = "ball"
                object_tracking["ball"] = True
            else:
                object_tracking["player_id"] = object_to_player.get(p, pd.NA)
                object_tracking["object_id"] = p
                object_tracking["ball"] = False

            object_tracking["x"] = tracking[f"{p}_x"].values.round(2)
            object_tracking["y"] = tracking[f"{p}_y"].values.round(2)
            object_tracking["z"] = tracking["ball_z"].values.round(2) if p == "ball" else np.nan

            for period_id in object_tracking["period_id"].unique():
                period_tracking = object_tracking[object_tracking["period_id"] == period_id].dropna(subset=["x"]).copy()
                if not period_tracking.empty:
                    period_tracking = self._calculate_object_kinematics(
                        period_tracking,
                        fps=fps,
                        is_ball=(p == "ball"),
                        apply_kinematic_correction=apply_kinematic_correction,
                    )
                    tracking_list.append(period_tracking)

        tracking_data = pd.concat(tracking_list, ignore_index=True)
        match_id = pd.NA
        if isinstance(getattr(self, "match_metadata", None), dict):
            raw_match_id = self.match_metadata.get("match_id", pd.NA)
            if pd.notna(raw_match_id):
                match_id = str(raw_match_id)
        tracking_data["match_id"] = match_id

        dtype_map = {
            "z": float,
            "vx": float,
            "vy": float,
            "ax": float,
            "ay": float,
            "speed": float,
            "accel_s": float,
            "accel_v": float,
        }
        raw_tracking = tracking_data.reset_index(drop=True).astype(dtype_map)
        input_tracking = tracking_data[tracking_data["ball_state"] == "alive"].reset_index(drop=True).astype(dtype_map)

        raw_tracking["period_id"] = raw_tracking["period_id"].map(CDF_PERIOD_MAP)
        raw_tracking["match_id"] = raw_tracking["match_id"].astype("string")
        raw_tracking["player_id"] = raw_tracking["player_id"].astype("string")
        raw_tracking["object_id"] = raw_tracking["object_id"].astype("string")

        input_tracking["period_id"] = input_tracking["period_id"].map(CDF_PERIOD_MAP)
        input_tracking["match_id"] = input_tracking["match_id"].astype("string")
        input_tracking["player_id"] = input_tracking["player_id"].astype("string")
        input_tracking["object_id"] = input_tracking["object_id"].astype("string")

        cdf_rename = {
            "period_id": "period",
            "ball_state": "ball_status",
            "ball_owning_team_id": "ball_poss_team_id",
            "speed": "vel",
            "accel_s": "acc_s",
            "accel_v": "acc",
        }
        ball_status_map = {"alive": True, "dead": False}

        raw_tracking = raw_tracking.rename(columns=cdf_rename)
        raw_tracking["ball_status"] = raw_tracking["ball_status"].map(ball_status_map)

        input_tracking = input_tracking.rename(columns=cdf_rename)
        input_tracking["ball_status"] = input_tracking["ball_status"].map(ball_status_map)

        ordered_cols = ["match_id"] + [col for col in raw_tracking.columns if col != "match_id"]
        raw_tracking = raw_tracking.loc[:, ordered_cols]
        input_tracking = input_tracking.loc[:, ordered_cols]

        return raw_tracking, input_tracking

    def preprocess_tracking_data(self, apply_kinematic_correction: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        tracking = self.tracking.copy()

        if "frame_id" not in tracking.columns or "utc_timestamp" not in tracking.columns:
            tracking = BaseEventTrackingPreprocessor.calculate_tracking_datetimes(self.events, tracking, self.fps)

        return self._finalize_tracking_output(
            tracking,
            fps=self.fps,
            apply_kinematic_correction=apply_kinematic_correction,
        )
    
    @staticmethod
    def merge_events_and_tracking(events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False) -> pd.DataFrame:
        events = events.copy()

        if "start_x" in events.columns:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "start_x", "start_y"]
        else:
            event_cols = ["period_id", "timestamp", "object_id", "spadl_type", "coordinates_x", "coordinates_y"]

        renamed_cols = ["period_id", "timestamp", "player_id", "event_type", "event_x", "event_y"]
        column_dict = dict(zip(event_cols, renamed_cols))

        events["timestamp"] = ((events["timestamp"] * fps).round().astype(int) / fps).round(2)
        merged = pd.merge(tracking, events[event_cols], how="left").rename(columns=column_dict)

        if ffill:
            merged[renamed_cols[2:]] = merged[renamed_cols[2:]].ffill()

        return merged

    def merge_synced_events_and_tracking(
        events: pd.DataFrame, tracking: pd.DataFrame, fps=25, ffill=False
    ) -> pd.DataFrame:
        assert "synced_ts" in events.columns

        column_mapping = {"spadl_type": "event_type", "start_x": "annot_x", "start_y": "annot_y"}
        events = events.copy().rename(columns=column_mapping)

        synced_cols = ["period_id", "synced_ts", "player_id", "event_type"]
        synced_events = events.loc[events["synced_ts"].notna(), synced_cols].copy().reset_index(drop=True)
        synced_events["timestamp"] = synced_events["synced_ts"].apply(timestamp_to_seconds)
        synced_events.drop("synced_ts", axis=1, inplace=True)

        annot_cols = ["period_id", "utc_timestamp", "annot_x", "annot_y"]
        annot_events = events[annot_cols].copy()
        annot_events = BaseEventTrackingPreprocessor.calculate_event_seconds(annot_events)
        annot_events["timestamp"] = ((annot_events["timestamp"] * fps).round().astype(int) / fps).round(2)
        annot_events.drop("utc_timestamp", axis=1, inplace=True)

        merged = pd.merge(tracking, synced_events, how="left")
        merged = pd.merge(merged, annot_events, how="left")

        event_mask = merged["player_id"].notna()
        merged.loc[event_mask, "event_x"] = merged.loc[event_mask, "ball_x"]
        merged.loc[event_mask, "event_y"] = merged.loc[event_mask, "ball_y"]

        if ffill:
            ffill_cols = ["player_id", "event_type", "event_x", "event_y", "annot_x", "annot_y"]
            merged[ffill_cols] = merged[ffill_cols].ffill()

        return merged
