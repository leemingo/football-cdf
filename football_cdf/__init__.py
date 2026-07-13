"""Provider-specific raw event/tracking preprocessing utilities.

Provider classes are imported lazily so event-only users do not need optional
tracking dependencies such as Kloppy until they request the Sportec adapter.
"""

from .skillcorner_paths import find_match_dir
from .statsbomb_paths import resolve_statsbomb_data_root, resolve_statsbomb_match_paths

__all__ = [
    "BeproDataPreprocessor",
    "SkillcornerDataPreprocessor",
    "SportecDataPreprocessor",
    "StatsbombDataPreprocessor",
    "find_match_dir",
    "resolve_statsbomb_data_root",
    "resolve_statsbomb_match_paths",
]


def __getattr__(name: str):
    if name == "BeproDataPreprocessor":
        from .bepro_preprocessing import BeproDataPreprocessor

        return BeproDataPreprocessor
    if name == "SkillcornerDataPreprocessor":
        from .skillcorner_preprocessing import SkillcornerDataPreprocessor

        return SkillcornerDataPreprocessor
    if name == "SportecDataPreprocessor":
        from .sportec_preprocessing import SportecDataPreprocessor

        return SportecDataPreprocessor
    if name == "StatsbombDataPreprocessor":
        from .statsbomb_preprocessing import StatsbombDataPreprocessor

        return StatsbombDataPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
