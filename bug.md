# Known Bugs & Fixes

## 1. Zero Volume in Tick Data

**Symptom**: `Volume` column in chart/features is 0.
**Cause**: MT5 Tick data often reports 0 for "Real Volume".
**Fix**: Use `tick_count` as Volume.
**Implementation**: Patched `train_pipeline.py` to check `if volume.sum() == 0` then `volume = tick_count`.

## 2. Duplicate Column Error

**Symptom**: `polars.exceptions.DuplicateError: column with name 'sr_swing_high' ...`
**Cause**: Calling `add_support_resistance` multiple times or joining tables with overlapping columns.
**Fix**: Rename intermediate columns inside features (e.g., `_tmp_swing`) or check existence (`if col not in df.columns`).

## 3. XGBoost 3.0 Compatibility

**Symptom**: `TypeError: Loss of type mixin to define estimator type`
**Cause**: `XGBClassifier` wrapper in `sklearn` pipeline is incompatible with new XGBoost 3.0 internals.
**Fix**: Use Native API (`xgb.train`) with `xgb.DMatrix` instead of the Classifier wrapper.
