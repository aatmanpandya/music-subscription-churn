import numpy as np
import pandas as pd
import pytest

from pipeline.schema import SchemaValidationError, validate_features
from tests.conftest import make_frame


def good(n=2000):
    return make_frame(n, seed=7)


def test_valid_frame_passes():
    validate_features(good(), require_label=True, require_unique_ids=True)


def test_missing_column_is_caught():
    with pytest.raises(SchemaValidationError, match="missing required columns.*auto_renew"):
        validate_features(good().drop(columns=["auto_renew"]))


def test_wrong_dtype_is_caught():
    df = good()
    df["hours_30"] = df["hours_30"].astype(str)
    with pytest.raises(SchemaValidationError, match="hours_30: expected numeric dtype"):
        validate_features(df)


def test_out_of_range_is_caught():
    df = good()
    df.loc[0, "hours_30"] = 900.0        # more than 24h * 30d
    df.loc[1, "completion_rate_30"] = 1.7
    with pytest.raises(SchemaValidationError) as e:
        validate_features(df)
    msg = str(e.value)
    assert "hours_30" in msg and "completion_rate_30" in msg      # both reported at once


def test_binary_flag_and_integer_checks():
    df = good()
    df.loc[0, "auto_renew"] = 7.0
    df.loc[1, "plan_days"] = 30.5
    with pytest.raises(SchemaValidationError) as e:
        validate_features(df)
    assert "auto_renew" in str(e.value) and "plan_days" in str(e.value)


def test_null_rate_limit_is_caught_on_batches():
    df = good(2000)
    df.loc[df.index[:600], "plan_days"] = np.nan        # 30% nulls, limit is 1%
    with pytest.raises(SchemaValidationError, match="plan_days: null rate"):
        validate_features(df)


def test_columns_that_must_never_be_null():
    df = good(50)                                        # tiny batch: statistical checks skipped...
    df.loc[0, "active_days_30"] = np.nan                 # ...but a must-fill column still fails
    with pytest.raises(SchemaValidationError, match="active_days_30: must not be null"):
        validate_features(df)


def test_infinite_values_caught():
    df = good()
    df.loc[3, "discount"] = np.inf
    with pytest.raises(SchemaValidationError, match="discount: contains infinite"):
        validate_features(df)


def test_bad_gender_and_labels_and_duplicate_ids():
    df = good()
    df.loc[0, "gender"] = "robot"
    df.loc[1, "is_churn"] = 3
    df.loc[2, "msno"] = df.loc[3, "msno"]
    with pytest.raises(SchemaValidationError) as e:
        validate_features(df, require_label=True, require_unique_ids=True)
    msg = str(e.value)
    assert "gender" in msg and "is_churn" in msg and "duplicate" in msg


def test_nullable_int_parquet_dtypes_accepted():
    """DuckDB-written parquet gives nullable Int64; that must be valid."""
    df = good()
    for c in ["plan_days", "auto_renew", "city"]:
        df[c] = df[c].astype("Int64")
    validate_features(df)


def test_zero_rows_rejected():
    with pytest.raises(SchemaValidationError, match="zero rows"):
        validate_features(good().iloc[:0])
