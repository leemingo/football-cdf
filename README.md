# football-cdf

Shared preprocessing that converts provider raw event + tracking feeds into a
**common data format (CDF)**. Single source of truth, consumed by multiple
projects (e.g. `player-aware-epv`, `football-analytics`) as a **git submodule**
+ editable install.

## Providers
- **SkillCorner** — `football_cdf.skillcorner_preprocessing.SkillcornerDataPreprocessor` (+ `skillcorner_api`)
- **Bepro** — `football_cdf.bepro_preprocessing.BeproDataPreprocessor` (+ `bepro_actions`)
- **Sportec / DFL** — `football_cdf.sportec_preprocessing.SportecDataPreprocessor`
- `base.BaseEventTrackingPreprocessor` (shared CDF logic), `constants` (`CDF_PERIOD_MAP`, `PITCH_X/Y`)

## Install (editable — shared across projects)
```bash
pip install -e /path/to/football-cdf
python -c "from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor; print('ok')"
```

## Use as a submodule in another repo
```bash
git submodule add <football-cdf remote/url> football-cdf
pip install -e ./football-cdf
# clone later:  git clone --recurse-submodules <superproject>
# update:       git submodule update --remote football-cdf
```

Edit the code **here** (in any project's `football-cdf/` submodule checkout),
commit + push, then `git submodule update --remote football-cdf` in the other
projects to pull the change.
