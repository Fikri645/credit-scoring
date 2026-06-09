"""Unit tests for the Polars feature-engineering primitives (no data needed)."""
import polars as pl

from src import config
from src.features import aggregate_depth, dates_to_relative, prune_columns


def test_aggregate_depth_one_row_per_case():
    df = pl.DataFrame({
        config.CASE_ID: [1, 1, 2, 2, 2],
        "num_group1": [0, 1, 0, 1, 2],
        "amount_A": [10.0, 20.0, 5.0, None, 15.0],
        "cat_M": ["a", "b", "a", "a", "c"],
    })
    out = aggregate_depth(df)
    assert out.height == 2                       # one row per case_id
    assert "amount_A_max" in out.columns
    assert "amount_A_mean" in out.columns
    assert "cat_M_nunique" in out.columns
    assert "num_records" in out.columns
    rec = out.sort(config.CASE_ID)["num_records"].to_list()
    assert rec == [2, 3]


def test_dates_to_relative_drops_raw_date():
    df = pl.DataFrame({
        config.CASE_ID: [1, 2],
        config.DATE_DECISION: pl.Series(["2020-01-10", "2020-02-20"]).str.to_date(),
        "event_D": pl.Series(["2020-01-01", "2020-02-10"]).str.to_date(),
    })
    out = dates_to_relative(df)
    assert config.DATE_DECISION not in out.columns
    assert "month_decision" in out.columns
    # event_D becomes a negative #days (event is before decision).
    assert out["event_D"].to_list() == [-9.0, -10.0]


def test_prune_drops_constant_and_protects():
    df = pl.DataFrame({
        "const": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "useful": list(range(10)),
        config.TARGET: [0, 1] * 5,
    })
    out = prune_columns(df, protect=(config.TARGET,))
    assert "const" not in out.columns
    assert "useful" in out.columns
    assert config.TARGET in out.columns
