"""StatsBomb labels mapped to the project's CDF-aligned conventions."""
from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd


STATSBOMB_POSITION_MAPPING = {
    "Goalkeeper": "GK",
    "Right Back": "RB",
    "Right Center Back": "RCB",
    "Center Back": "CB",
    "Left Center Back": "LCB",
    "Left Back": "LB",
    "Right Wing Back": "RWB",
    "Left Wing Back": "LWB",
    "Right Defensive Midfield": "RDM",
    "Center Defensive Midfield": "CDM",
    "Left Defensive Midfield": "LDM",
    "Right Midfield": "RM",
    "Right Center Midfield": "RCM",
    "Center Midfield": "CM",
    "Left Center Midfield": "LCM",
    "Left Midfield": "LM",
    "Right Wing": "RWF",
    "Right Attacking Midfield": "RAM",
    "Center Attacking Midfield": "CAM",
    "Left Attacking Midfield": "LAM",
    "Left Wing": "LWF",
    "Right Center Forward": "RCF",
    "Striker": "CF",
    "Center Forward": "CF",
    "Left Center Forward": "LCF",
    "Secondary Striker": "CF",
}

EVENT_TYPE_MAPPING = {
    "Ball Receipt*": "ball_receipt",
    "Ball Recovery": "ball_recovery",
    "Dispossessed": "dispossessed",
    "Duel": "duel",
    "Camera On": "camera_on",
    "Block": "block",
    "Offside": "offside",
    "Clearance": "clearance",
    "Interception": "interception",
    "Dribble": "dribble",
    "Shot": "shot",
    "Pressure": "pressure",
    "Half Start": "half_start",
    "Substitution": "substitution",
    "Own Goal Against": "own_goal_against",
    "Foul Won": "foul_won",
    "Foul Committed": "foul_committed",
    "Goal Keeper": "goalkeeper",
    "Bad Behaviour": "bad_behaviour",
    "Own Goal For": "own_goal_for",
    "Player On": "player_on",
    "Player Off": "player_off",
    "Shield": "shield",
    "Starting XI": "starting_xi",
    "Half End": "half_end",
    "Referee Ball-Drop": "referee_ball_drop",
    "Error": "error",
    "Miscontrol": "miscontrol",
    "Dribbled Past": "dribbled_past",
    "Injury Stoppage": "injury_stoppage",
    "Tactical Shift": "tactical_shift",
    "50/50": "50_50",
    "Carry": "carry",
    "Pass": "pass",
}

PASS_SUB_TYPE_MAPPING = {
    "Corner": "corner_kick",
    "Free Kick": "free_kick",
    "Goal Kick": "goal_kick",
    "Interception": "interception",
    "Kick Off": "kick_off",
    "Recovery": "recovery",
    "Throw-in": "throw_in",
}

SHOT_SUB_TYPE_MAPPING = {
    "Free Kick": "free_kick",
    "Open Play": "open_play",
    "Penalty": "penalty",
}


def nested_name(value: object) -> object:
    """Return a StatsBomb ``{id, name}`` object's name."""
    if isinstance(value, Mapping):
        return value.get("name", pd.NA)
    return pd.NA


def normalize_label(value: object) -> object:
    """Normalize a provider label to snake_case without losing numeric labels."""
    if value is None or pd.isna(value):
        return pd.NA
    normalized = str(value).strip()
    if not normalized:
        return pd.NA
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = normalized.replace("*", "").replace("-", "_").replace(" ", "_").replace("/", "_")
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower() or pd.NA


def derive_cdf_type(event: Mapping) -> object:
    """Return the project CDF event type for one raw StatsBomb event."""
    raw_type = nested_name(event.get("type"))
    if raw_type == "Pass" and bool((event.get("pass") or {}).get("cross", False)):
        return "cross"
    return EVENT_TYPE_MAPPING.get(raw_type, normalize_label(raw_type))


def derive_cdf_sub_type(event: Mapping) -> object:
    """Return a normalized event subtype from the type-specific object."""
    raw_type = nested_name(event.get("type"))
    if raw_type == "Pass":
        value = nested_name((event.get("pass") or {}).get("type"))
        return PASS_SUB_TYPE_MAPPING.get(value, normalize_label(value))
    if raw_type == "Shot":
        value = nested_name((event.get("shot") or {}).get("type"))
        return SHOT_SUB_TYPE_MAPPING.get(value, normalize_label(value))
    if raw_type == "Duel":
        return normalize_label(nested_name((event.get("duel") or {}).get("type")))
    if raw_type == "Goal Keeper":
        return normalize_label(nested_name((event.get("goalkeeper") or {}).get("type")))
    if raw_type == "Foul Committed":
        foul = event.get("foul_committed") or {}
        value = nested_name(foul.get("type"))
        if pd.isna(value):
            value = nested_name(foul.get("card"))
        return normalize_label(value)
    if raw_type == "Substitution":
        return normalize_label(nested_name((event.get("substitution") or {}).get("outcome")))
    return pd.NA


def _raw_outcome_name(event: Mapping) -> object:
    raw_type = nested_name(event.get("type"))
    field = {
        "Pass": "pass",
        "Shot": "shot",
        "Dribble": "dribble",
        "Duel": "duel",
        "Interception": "interception",
        "Goal Keeper": "goalkeeper",
        "50/50": "50_50",
        "Ball Receipt*": "ball_receipt",
        "Substitution": "substitution",
    }.get(raw_type)
    if field is None:
        return pd.NA
    payload = event.get(field) or {}
    outcome = nested_name(payload.get("outcome"))
    if pd.isna(outcome) and raw_type == "Duel":
        outcome = nested_name(payload.get("type"))
    if pd.isna(outcome) and raw_type == "Goal Keeper":
        outcome = nested_name(payload.get("type"))
    return outcome


def infer_cdf_outcome(event: Mapping) -> tuple[object, object, object]:
    """Return ``(outcome, outcome_detailed, success)`` for one event."""
    raw_type = nested_name(event.get("type"))
    raw_outcome = _raw_outcome_name(event)
    detailed = normalize_label(raw_outcome)

    if raw_type in {"Pass", "Ball Receipt*"}:
        if pd.isna(raw_outcome):
            return "successful", pd.NA, True
        if "Offside" in str(raw_outcome):
            return "offside", detailed, False
        return "unsuccessful", detailed, False

    if raw_type == "Carry":
        return "successful", pd.NA, True

    if raw_type == "Shot":
        success = raw_outcome == "Goal"
        return ("successful" if success else "unsuccessful"), detailed, success

    if raw_type == "Dribble":
        success = raw_outcome == "Complete"
        return ("successful" if success else "unsuccessful"), detailed, success

    if raw_type == "Ball Recovery":
        failed = bool((event.get("ball_recovery") or {}).get("recovery_failure", False))
        return (
            ("unsuccessful" if failed else "successful"),
            ("recovery_failure" if failed else pd.NA),
            not failed,
        )

    if raw_type in {"Miscontrol", "Dispossessed", "Error", "Own Goal Against"}:
        return "unsuccessful", detailed, False

    if raw_type == "Foul Committed":
        return "unsuccessful", detailed, False

    if raw_type in {"Block", "Shield"}:
        return "successful", detailed, True

    if raw_type in {"Duel", "Interception", "50/50"} and pd.notna(raw_outcome):
        label = str(raw_outcome).lower()
        success = any(token in label for token in ("won", "success"))
        failure = any(token in label for token in ("lost", "failure"))
        if success or failure:
            return ("successful" if success else "unsuccessful"), detailed, success

    if raw_type == "Goal Keeper" and pd.notna(raw_outcome):
        goalkeeper_type = nested_name((event.get("goalkeeper") or {}).get("type"))
        if goalkeeper_type == "Goal Conceded":
            return "unsuccessful", detailed, False
        if goalkeeper_type == "Shot Faced":
            return "neutral", detailed, pd.NA
        label = str(raw_outcome).lower()
        failure = any(token in label for token in ("goal conceded", "lost", "failure"))
        return ("unsuccessful" if failure else "successful"), detailed, not failure

    neutral_types = {
        "Starting XI",
        "Tactical Shift",
        "Half Start",
        "Half End",
        "Substitution",
        "Pressure",
        "Foul Won",
        "Dribbled Past",
        "Injury Stoppage",
        "Referee Ball-Drop",
        "Offside",
        "Own Goal For",
    }
    if raw_type in neutral_types:
        return "neutral", detailed, pd.NA

    return pd.NA, detailed, pd.NA


def infer_body_part(event: Mapping) -> object:
    """Return the normalized CDF body-part name."""
    raw_type = nested_name(event.get("type"))
    field = {
        "Pass": "pass",
        "Shot": "shot",
        "Clearance": "clearance",
        "Goal Keeper": "goalkeeper",
    }.get(raw_type)
    if field is None:
        return pd.NA
    value = nested_name((event.get(field) or {}).get("body_part"))
    mapping = {
        "Left Foot": "left_foot",
        "Right Foot": "right_foot",
        "Head": "head",
        "Other": "other",
        "Drop Kick": "other",
        "Keeper Arm": "other",
        "Both Hands": "other",
        "Left Hand": "other",
        "Right Hand": "other",
        "Chest": "other",
    }
    return mapping.get(value, normalize_label(value))


def normalize_position(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    return STATSBOMB_POSITION_MAPPING.get(str(value), str(value))
