# football-cdf

`football-cdf` contains shared preprocessing utilities for converting provider
raw event and tracking feeds into the **Common Data Format (CDF)** described in
[Anzer et al., "Common Data Format (CDF): A Standardized Format for Match-Data
in Football (Soccer)"](https://arxiv.org/abs/2505.15820).
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
| SkillCorner | `football_cdf.skillcorner_preprocessing.SkillcornerDataPreprocessor` | Metadata, lineup, Dynamic Events helpers, and tracking JSONL parsing. |
| Sportec / DFL | `football_cdf.sportec_preprocessing.SportecDataPreprocessor` | Event conversion, Kloppy-backed tracking loading/normalization, and SPADL-style action conversion for already-extracted raw folders. |
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

## Install

When working from the parent `open-football-analytics` repository, install the
parent package in editable mode:

```bash
pip install -e ".[models,notebooks]"
```

When using `football-cdf` independently:

```bash
pip install -e /path/to/football-cdf
python -c "from football_cdf.skillcorner_preprocessing import SkillcornerDataPreprocessor; print('ok')"
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
