# football-cdf

`football-cdf` contains shared preprocessing utilities for converting provider
raw event and tracking feeds into this project's **CDF-aligned canonical
tables**. These flat pandas tables are inspired by the Common Data Format (CDF)
described in [Anzer et al., "Common Data Format (CDF): A Standardized Format
for Match-Data in Football (Soccer)"](https://arxiv.org/abs/2505.15820), but
they are not the official nested CDF 0.2.x JSON/JSONL schemas and are not
claimed to pass an external CDF validator. The explicit column contracts live
in `football_cdf/contracts.py`.

For action-level event tables, provider events are mapped to
[SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/SPADL_definitions.html)
action conventions where the provider parser supports it.
Sportec / DFL tracking is loaded through
[Kloppy](https://kloppy.pysport.org/), which provides the provider parser,
pitch-dimension normalization, and orientation transform before conversion into
the CDF-style wide tracking table.

In `open-football-analytics`, this submodule provides the preprocessing layer
used before metric workflows such as xG, xPass, and xT.

## Provider Support

| Provider | Main class / module | Notes |
|---|---|---|
| Bepro | `football_cdf.bepro_preprocessing.BeproDataPreprocessor` | Metadata, lineup, provider event conversion, tracking where supplied, and SPADL-style actions. |
| SkillCorner | `football_cdf.skillcorner_preprocessing.SkillcornerDataPreprocessor` | Metadata, lineup, Dynamic Events helpers, and tracking JSONL parsing. |
| Sportec / DFL | `football_cdf.sportec_preprocessing.SportecDataPreprocessor` | Event conversion, Kloppy-backed tracking loading/normalization, and SPADL-style action conversion for already-extracted raw folders. |
| StatsBomb Open Data | `football_cdf.statsbomb_preprocessing.StatsbombDataPreprocessor` | Metadata, lineup, CDF-aligned events, SPADL-style actions, and event-linked 360 context. StatsBomb 360 is not treated as continuous tracking. |
| Shared CDF logic | `football_cdf.base.BaseEventTrackingPreprocessor` | Common event/tracking schema helpers. |

## Public Tutorial

The public notebook is:

```text
notebooks/provider_to_cdf.ipynb
```

It demonstrates:

- SkillCorner Open Data file layout
- conversion into the match-bundle layout expected by
  `SkillcornerDataPreprocessor`
- SkillCorner tracking JSONL to wide CDF tracking
- Sportec raw folder loading when the Sportec files are already extracted
- Sportec tracking loading and coordinate/orientation normalization through
  Kloppy
- Sportec event conversion into SPADL-style action rows for action-sequence
  inspection
- StatsBomb event/CDF/SPADL conversion and an event-linked 360 freeze frame
- Bepro v1 event, tracking, CDF, and SPADL conversion using the same plotting
  and action-sequence helpers
- explicit Bepro contract and data-quality checks that fail the notebook when
  identifiers, coordinates, scores, action references, or tracking schema are
  inconsistent

The notebook intentionally uses placeholder paths and public-data assumptions.

## SkillCorner Open Data Layout

The tutorial expects the SkillCorner Open Data repository layout:

```text
path/to/SkillCorner/opendata/data/matches/{match_id}/
  {match_id}_match.json
  {match_id}_dynamic_events.csv
  {match_id}_tracking_extrapolated.jsonl
```

The helper in the notebook creates a normalized match bundle:

```text
/path/to/cdf-work/{match_id}/
  match.json
  dynamic_events.csv
  tracking.jsonl
```

## Sportec Layout

Sportec examples assume the raw files are already extracted:

```text
path/to/dfl-spoho/raw/{match_id}/
```

Point `SPORTEC_ROOT` at the raw root and `SPORTEC_MATCH` at the match folder
name.

## StatsBomb Open Data

Point the preprocessor at either a local clone of
[`statsbomb/open-data`](https://github.com/statsbomb/open-data) or its `data/`
directory. Match metadata is indexed from the official repository layout and
the optional 360 file is discovered automatically:

```text
path/to/open-data/data/
  competitions.json
  matches/{competition_id}/{season_id}.json
  events/{match_id}.json
  lineups/{match_id}.json
  three-sixty/{match_id}.json  # optional
```

```python
from football_cdf import StatsbombDataPreprocessor

preprocessor = StatsbombDataPreprocessor(
    "/path/to/open-data",
    match_id="3857276",
    load_360=True,
)

metadata = preprocessor.get_match_metadata(mode="full")
lineup = preprocessor.get_lineup(mode="full")
events_cdf = preprocessor.preprocess_cdf_events(preserve_raw_columns=True)
actions = preprocessor.preprocess_spadl_events(events_cdf)
frames_360, objects_360 = preprocessor.preprocess_360_data()
```

StatsBomb's 120 x 80 coordinates are converted to the project's centered
105 x 68 pitch. Because StatsBomb event coordinates are oriented left-to-right
for the team executing the action, away-team coordinates are rotated to create
the same static home-left frame used by the other preprocessors. Match clock
values are preserved, while `utc_timestamp` is `NaT` because Open Data kickoff
times do not include a reliable timezone. The 360 tables remain event-linked
snapshots: non-actor player identities are deliberately left missing and
observations outside the nominal pitch are preserved with `in_pitch=False`.

By default, the StatsBomb action conversion keeps the same useful extended
action types as the existing providers (`ball_recovery`, `dispossessed`, and
`shot_block`). Set `include_extended_actions=False` for the core SPADL-style
vocabulary. A StatsBomb interception pass is split into interception and pass
actions; set `split_interception_passes=False` to keep only its pass action.
No synthetic between-event dribbles are inserted because StatsBomb already
provides explicit `Carry` events.

## Install

When working from the parent `open-football-analytics` repository, install the
parent package in editable mode:

```bash
pip install -e ".[models,notebooks]"
```

When using `football-cdf` independently:

```bash
pip install -e /path/to/football-cdf
python -c "from football_cdf import StatsbombDataPreprocessor; print('ok')"
```

## Use As A Submodule

```bash
git submodule add <football-cdf remote/url> football-cdf
pip install -e ./football-cdf
```

Clone later with:

```bash
git clone --recurse-submodules <superproject>
```

Update the submodule with:

```bash
git submodule update --remote football-cdf
```
