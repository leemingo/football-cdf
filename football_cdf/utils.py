"""Small utilities shared by preprocessing modules."""


def timestamp_to_seconds(timestamp: str) -> float:
    minutes, seconds = timestamp.split(":")
    return float(minutes) * 60 + float(seconds)
