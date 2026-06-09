"""Shared constants for raw event and tracking preprocessing."""

PITCH_X = 105.0
PITCH_Y = 68.0

DEFAULT_FPS = 25

CDF_PERIOD_MAP = {
    1: "first_half",
    2: "second_half",
    3: "first_half_extra",
    4: "second_half_extra",
    5: "penalty_shootout",
}

KINEMATIC_CORRECTION_DEFAULTS = {
    "player": {"window_length": 7, "polyorder": 1, "max_speed": 12.0, "max_acceleration": 6.0},
    "ball": {"window_length": 3, "polyorder": 1, "max_speed": 28.0, "max_acceleration": 13.5},
}

SMOOTHING_DEFAULTS = {
    "velocity": {"window_length": 15, "polyorder": 2},
    "acceleration": {"window_length": 9, "polyorder": 2},
}

POSITION_MAPPING = {
    None: None,
    "TW": "GK",
    "IVR": "RCB",
    "IVL": "LCB",
    "IVZ": "CB",
    "RV": "RB",
    "LV": "LB",
    "DMR": "RDM",
    "DRM": "RDM",
    "DML": "LDM",
    "DLM": "LDM",
    "DMZ": "CDM",
    "HR": "RCM",
    "HL": "LCM",
    "MZ": "CM",
    "RM": "RM",
    "LM": "LM",
    "ORM": "RAM",
    "OHR": "RAM",
    "OLM": "LAM",
    "OHL": "LAM",
    "ZO": "CAM",
    "RA": "RWF",
    "LA": "LWF",
    "HST": "CF",
    "STR": "RCF",
    "STL": "LCF",
    "STZ": "CF",
}
