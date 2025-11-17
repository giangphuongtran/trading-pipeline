# Bugs and Fixes Documentation

This document tracks all bugs discovered and fixed during the development of the trading pipeline project.

## Table of Contents
- [Data Quality Issues](#data-quality-issues)
- [Code Bugs](#code-bugs)
- [Configuration Issues](#configuration-issues)
- [Known Limitations](#known-limitations)

---

## Data Quality Issues

### 1. Intraday Timestamp Gaps Detection

**Issue**: The original gap detection only flagged rows with gaps but didn't identify the specific missing timestamps. For example, if row 374 had timestamp `09:05:00` and row 375 had `09:15:00`, the system would flag row 375 but wouldn't tell you that `09:10:00` was missing.

**Impact**: Made it impossible to know exactly which timestamps needed to be backfilled from the API.

**Fix Date**: 2024-01-XX

**Solution**: Enhanced `_warn_if_timestamp_gaps()` function in `ml/scripts/prepare_features.py` to:
- Generate a list of all missing timestamps between gaps
- Add `missing_timestamps` column to the gap warnings DataFrame
- Add `missing_count` to show how many bars are missing
- Improve reporting to show first 10 missing timestamps in console output

**Code Changes**:
- Modified `_warn_if_timestamp_gaps()` to accept `expected_interval_minutes` parameter (default: 5)
- Added logic to generate expected timestamps in gaps
- Updated return DataFrame schema to include `missing_timestamps` and `missing_count` columns

**Example Output**:
```
⚠️  AAPL: detected 3 intraday gaps (>7.5 minutes) with 5 missing bars. 
Missing timestamps: [Timestamp('2023-11-13 09:10:00+00:00'), ...]
```

---

### 2. Daily Bar Gaps - Holiday vs Data Gap Confusion

**Issue**: The system flagged 4-day gaps (e.g., Friday to Tuesday) as data quality issues, but these are often legitimate market holidays (Christmas, New Year's Day, etc.) where the market is closed.

**Impact**: False positives in data quality checks, making it hard to identify real data gaps.

**Status**: **Documented but not yet implemented** - Holiday detection logic needs to be added.

**Proposed Solution**: 
- Use `pandas_market_calendars` library to check if gaps contain trading days
- Maintain a list of known market holidays
- Add `is_holiday_gap` flag to gap warnings DataFrame
- Filter out holiday gaps from data quality reports

**Known Market Holidays (2023-2025)**:
See [Market Holidays](#market-holidays) section below.

---

## Code Bugs

### 3. Dataclass Mutable Default Argument Error

**Issue**: `RuleBasedSentimentModel` in `ml/features/sentiment_models.py` used mutable default arguments (sets) for `positive_words` and `negative_words`, causing:
```
ValueError: mutable default <class 'set'> for field positive_words is not allowed: use default_factory
```

**Impact**: Import errors when loading the `ml.features` module.

**Fix Date**: 2024-01-XX

**Solution**: Changed from:
```python
positive_words: Iterable[str] = POSITIVE_WORDS
negative_words: Iterable[str] = NEGATIVE_WORDS
```

To:
```python
positive_words: Iterable[str] = field(default_factory=lambda: set(POSITIVE_WORDS))
negative_words: Iterable[str] = field(default_factory=lambda: set(NEGATIVE_WORDS))
```

**Code Location**: `ml/features/sentiment_models.py`

---

### 4. Missing Import in `ml/features/__init__.py`

**Issue**: `CandlestickFeatureEngineer` and `ConfluenceConfig` were not exported from `ml.features`, causing import errors in notebooks.

**Impact**: `ImportError: cannot import name 'CandlestickFeatureEngineer' from 'ml.features'`

**Fix Date**: 2024-01-XX

**Solution**: Added missing imports:
```python
from .candlestick_features import CandlestickFeatureEngineer
from .indicator_config import ConfluenceConfig
```

And added to `__all__` list.

**Code Location**: `ml/features/__init__.py`

---

### 5. Notebook `__file__` Attribute Error

**Issue**: Notebook cells tried to use `__file__` which doesn't exist in Jupyter notebooks, causing:
```
NameError: name '__file__' is not defined
```

**Impact**: Notebooks couldn't locate project root for imports.

**Fix Date**: 2024-01-XX

**Solution**: Added fallback logic:
```python
try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    ROOT = Path.cwd().resolve().parents[2]
```

**Code Location**: `ml/notebooks/full_feature_pipeline.ipynb`

**Note**: Later improved to use a `_locate_project_root()` helper function that searches for the project root by marker name.

---

### 6. Database Query Parameter Type Error

**Issue**: When passing a list of tickers to `_load_daily_bars()` or `_load_intraday_bars()`, psycopg2 was incorrectly formatting the parameter, causing:
```
DatabaseError: operator does not exist: character varying = text[]
LINE 4: WHERE ticker = ARRAY['AAPL']
```

**Impact**: Couldn't query multiple tickers at once.

**Fix Date**: 2024-01-XX

**Solution**: Created `_build_ticker_filter()` helper function that:
- Detects if ticker is a string, list, or None
- Uses `WHERE ticker = %s` for single ticker
- Uses `WHERE ticker = ANY(%s)` for multiple tickers
- Returns empty string for None (all tickers)

**Code Location**: `ml/scripts/prepare_features.py`

---

## Configuration Issues

### 7. Environment Variable Loading in Notebooks

**Issue**: `.env` file variables weren't automatically loaded when running notebooks, causing database connection failures.

**Impact**: Had to manually export environment variables before running notebooks.

**Fix Date**: 2024-01-XX

**Solution**: Added `python-dotenv` import and `load_dotenv()` call at the start of notebook setup cells.

**Code Location**: `ml/notebooks/full_feature_pipeline.ipynb`

---

### 8. Database Hostname Resolution

**Issue**: Using `local` as database hostname in `.env` file caused:
```
could not translate host name "local" to address: nodename nor servname provided, or not known
```

**Impact**: Database connection failures.

**Solution**: Changed to `localhost` or actual IP address in `.env` file.

**Note**: This is a configuration issue, not a code bug. Documented here for reference.

---

## Known Limitations

### 1. Holiday Gap Detection Not Implemented

**Status**: Documented but not yet implemented.

**Workaround**: Manually review daily gap warnings and filter out known holidays.

**Future Work**: Implement holiday calendar checking using `pandas_market_calendars`.

---

### 2. No Automatic Backfill for Missing Data

**Status**: Gap detection works, but no automatic backfill mechanism exists.

**Current Workflow**: 
1. Run feature preparation
2. Review gap warnings
3. Manually trigger backfill for missing dates/timestamps

**Future Work**: Create `identify_missing_data()` and `backfill_missing_data()` functions to automate this process.

---

## Market Holidays

The following are known market holidays that cause legitimate 4-day gaps (Friday to Tuesday) in daily bar data:

### 2023
- **Christmas Day + Weekend**: Dec 22 (Fri) → Dec 26 (Tue) - Market closed Dec 25 (Mon)
- **New Year's Day + Weekend**: Dec 29 (Fri) → Jan 2, 2024 (Tue) - Market closed Jan 1 (Mon)

### 2024
- **Martin Luther King, Jr. Day + Weekend**: Jan 12 (Fri) → Jan 16 (Tue) - Market closed Jan 15 (Mon)
- **Presidents' Day + Weekend**: Feb 16 (Fri) → Feb 20 (Tue) - Market closed Feb 19 (Mon)
- **Good Friday + Weekend**: Mar 28 (Thu) → Apr 1 (Mon) - Market closed Mar 29 (Fri)
- **Memorial Day + Weekend**: May 24 (Fri) → May 28 (Tue) - Market closed May 27 (Mon)
- **Labor Day + Weekend**: Aug 30 (Fri) → Sep 3 (Tue) - Market closed Sep 2 (Mon)

### 2025
- **Martin Luther King, Jr. Day + Weekend**: Jan 17 (Fri) → Jan 21 (Tue) - Market closed Jan 20 (Mon)
- **Presidents' Day + Weekend**: Feb 14 (Fri) → Feb 18 (Tue) - Market closed Feb 17 (Mon)
- **Good Friday + Weekend**: Apr 17 (Thu) → Apr 21 (Mon) - Market closed Apr 18 (Fri)
- **Memorial Day + Weekend**: May 23 (Fri) → May 27 (Tue) - Market closed May 26 (Mon)
- **Independence Day + Weekend**: Jul 3 (Thu) → Jul 7 (Mon) - Market closed Jul 4 (Fri)
- **Labor Day + Weekend**: Aug 29 (Fri) → Sep 2 (Tue) - Market closed Sep 1 (Mon)

**Note**: These holidays should be excluded from data quality gap warnings. A holiday calendar integration is planned for future releases.

---

## Testing Recommendations

After fixing bugs, always test:
1. ✅ Import all modules without errors
2. ✅ Run feature preparation pipeline end-to-end
3. ✅ Verify gap detection shows correct missing timestamps
4. ✅ Test with single ticker, multiple tickers, and None (all tickers)
5. ✅ Verify notebook cells run without `__file__` errors
6. ✅ Check that dataclass instances can be created without mutable default errors

---

## Related Documentation
- [Manual Backfill Commands](./manual_backfill_commands.md)
- [API Examples](../misc/API_EXAMPLE.md)
- [Project Thinking](../PROJECT_THINKING.md)

