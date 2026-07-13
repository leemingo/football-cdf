"""Column contracts for the project's CDF-aligned canonical tables.

These contracts describe the flat pandas tables used inside this repository.
They are inspired by Football CDF, but are not the official nested CDF JSON /
JSONL schemas.  Keeping the distinction explicit lets provider preprocessors
share one stable analytical interface without claiming validator compatibility.
"""

PROJECT_CANONICAL_VERSION = "v1"

EVENT_CDF_CORE_COLUMNS = (
    "match_id",
    "event_id",
    "event_index",
    "utc_timestamp",
    "match_clock",
    "match_clock_seconds",
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
)

EVENT_SPADL_HELPER_COLUMNS = (
    "success",
    "object_id",
    "receiver_object_id",
    "home_score",
    "away_score",
    "score",
)

SPADL_ACTION_COLUMNS = (
    "match_id",
    "original_event_id",
    "event_index",
    "period_id",
    "utc_timestamp",
    "match_clock",
    "time_seconds",
    "team_id",
    "player_id",
    "object_id",
    "receiver_id",
    "receiver_object_id",
    "spadl_type",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "bodypart_name",
    "result_name",
    "success",
    "home_score",
    "away_score",
    "score",
)

THREE_SIXTY_FRAME_COLUMNS = (
    "match_id",
    "event_id",
    "event_index",
    "actor_team_id",
    "actor_player_id",
    "visible_area",
    "n_visible_players",
    "has_actor",
)

THREE_SIXTY_OBJECT_COLUMNS = (
    "match_id",
    "event_id",
    "event_index",
    "observed_object_index",
    "team_id",
    "player_id",
    "teammate",
    "actor",
    "keeper",
    "x",
    "y",
    "in_pitch",
    "source_x",
    "source_y",
)
