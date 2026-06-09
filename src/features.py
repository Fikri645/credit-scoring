"""Feature engineering for the Home Credit 2024 *Credit Risk Model Stability*
dataset.

The competition ships a **depth-based relational schema**:

* ``base``  — one row per ``case_id`` (the loan application being scored),
  carrying ``date_decision``, ``WEEK_NUM`` and the ``target``.
* **depth-0** tables — static, one row per ``case_id`` (join directly).
* **depth-1** tables — one-to-many per ``case_id`` (indexed by ``num_group1``).
* **depth-2** tables — one-to-many-to-many (``num_group1`` + ``num_group2``).

Column names encode their transformation via a **trailing letter**:

==========  ====================================
suffix      meaning
==========  ====================================
``P``       DPD (days past due) transform
``A``       amount transform
``D``       date transform
``M``       masked categorical
``T``/``L`` other (numeric or categorical)
==========  ====================================

Strategy: aggregate every depth>0 table down to **one row per case_id**
(numeric → max/min/mean/var/last, dates → max/min/last, categoricals →
last/mode), then left-join everything onto ``base``. Date columns are
converted to *days relative to ``date_decision``* so the model never sees a
raw calendar date (which would leak the application period and destroy the
stability metric). All aggregation looks only at the rows belonging to a
``case_id`` — there is no cross-case leakage.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from src import config


# --------------------------------------------------------------------------- #
# Dtype handling
# --------------------------------------------------------------------------- #
def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Cast columns to memory-efficient dtypes based on the suffix convention."""
    for col in df.columns:
        if col in (config.CASE_ID, "WEEK_NUM", "num_group1", "num_group2"):
            df = df.with_columns(pl.col(col).cast(pl.Int64))
        elif col in (config.DATE_DECISION,) or col.endswith("D"):
            df = df.with_columns(pl.col(col).cast(pl.Date, strict=False))
        elif col.endswith(("P", "A")):
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        elif col.endswith(("M",)):
            df = df.with_columns(pl.col(col).cast(pl.String))
    return df


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in (
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )


# --------------------------------------------------------------------------- #
# Aggregation of depth>0 tables
# --------------------------------------------------------------------------- #
def aggregate_depth(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse a one-to-many depth table to one row per ``case_id``.

    Numeric → max/min/mean/var/last; date → max/min/last; string → last +
    n_unique. ``num_group*`` index columns are dropped after producing a row
    count so the model still sees *how many* records each applicant has.
    """
    group_cols = [config.CASE_ID]
    index_cols = {"num_group1", "num_group2"}

    exprs: list[pl.Expr] = [pl.len().alias("num_records")]
    for col in df.columns:
        if col in group_cols or col in index_cols:
            continue
        dtype = df.schema[col]
        if dtype == pl.Date:
            exprs += [
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).last().alias(f"{col}_last"),
            ]
        elif _is_numeric(dtype):
            exprs += [
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).var().alias(f"{col}_var"),
                pl.col(col).last().alias(f"{col}_last"),
            ]
        else:  # string / categorical
            exprs += [
                pl.col(col).drop_nulls().last().alias(f"{col}_last"),
                pl.col(col).n_unique().alias(f"{col}_nunique"),
            ]
    return df.group_by(group_cols).agg(exprs)


# --------------------------------------------------------------------------- #
# Date → relative-days transform (leakage guard)
# --------------------------------------------------------------------------- #
def dates_to_relative(df: pl.DataFrame) -> pl.DataFrame:
    """Replace every ``Date`` column with #days from ``date_decision``.

    Keeps ``month_decision`` / ``weekday_decision`` as mild seasonality
    signals, then drops the raw ``date_decision`` so no absolute calendar
    information (which is what the stability metric punishes) reaches the model.
    """
    if config.DATE_DECISION not in df.columns:
        return df

    df = df.with_columns(
        pl.col(config.DATE_DECISION).dt.month().alias("month_decision"),
        pl.col(config.DATE_DECISION).dt.weekday().alias("weekday_decision"),
    )
    date_cols = [c for c, t in df.schema.items()
                 if t == pl.Date and c != config.DATE_DECISION]
    for col in date_cols:
        df = df.with_columns(
            (pl.col(col) - pl.col(config.DATE_DECISION)).dt.total_days()
            .cast(pl.Float32).alias(col)
        )
    return df.drop(config.DATE_DECISION)


# --------------------------------------------------------------------------- #
# Column pruning — drop near-constant / mostly-null columns
# --------------------------------------------------------------------------- #
def prune_columns(
    df: pl.DataFrame,
    max_null_frac: float = 0.95,
    max_freq_frac: float = 0.95,
    protect: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Drop columns that are >95% null or >95% a single value (uninformative).

    ``protect`` columns (target, ids, week, protected attributes) are never
    dropped.
    """
    keep, n = [], len(df)
    for col in df.columns:
        if col in protect:
            keep.append(col)
            continue
        null_frac = df[col].null_count() / n
        if null_frac > max_null_frac:
            continue
        if _is_numeric(df.schema[col]) or df.schema[col] == pl.String:
            vc = df[col].drop_nulls().value_counts(sort=True)
            if len(vc) > 0 and vc[vc.columns[1]][0] / n > max_freq_frac:
                continue
        keep.append(col)
    return df.select(keep)


# --------------------------------------------------------------------------- #
# Pipeline driver
# --------------------------------------------------------------------------- #
def _depth_of(path: Path) -> int:
    """Infer table depth from the Kaggle filename convention ``*_<depth>[_n].parquet``."""
    stem = path.stem  # e.g. train_credit_bureau_a_2_0
    for part in reversed(stem.split("_")):
        if part.isdigit():
            return int(part) if int(part) <= 2 else 1
    return 0


def build_feature_table(split_dir: Path, is_train: bool = True) -> pl.DataFrame:
    """Read every parquet shard for a split, aggregate, and join onto base.

    Returns a single wide one-row-per-case_id Polars frame ready for modelling.
    """
    prefix = "train" if is_train else "test"
    base = pl.read_parquet(split_dir / f"{prefix}_base.parquet").pipe(set_table_dtypes)

    # Group shard files by their logical table name (strip the trailing _<n>).
    shards: dict[str, list[Path]] = {}
    for p in sorted(split_dir.glob(f"{prefix}_*.parquet")):
        if p.name == f"{prefix}_base.parquet":
            continue
        # logical name without the final numeric shard index
        parts = p.stem.split("_")
        logical = "_".join(parts[:-1]) if parts[-1].isdigit() else p.stem
        shards.setdefault(logical, []).append(p)

    feature_frames: list[pl.DataFrame] = []
    for logical, paths in shards.items():
        depth = _depth_of(paths[0])
        # Concatenate shards of the same logical table, then aggregate if needed.
        frames = [pl.read_parquet(p).pipe(set_table_dtypes) for p in paths]
        tbl = pl.concat(frames, how="diagonal_relaxed")
        if depth > 0:
            tbl = aggregate_depth(tbl)
        # Prefix non-key columns to avoid collisions across tables.
        rename = {c: f"{logical}__{c}" for c in tbl.columns if c != config.CASE_ID}
        feature_frames.append(tbl.rename(rename))

    df = base
    for fr in feature_frames:
        df = df.join(fr, on=config.CASE_ID, how="left")

    df = dates_to_relative(df)
    return df


if __name__ == "__main__":
    import time

    t0 = time.time()
    print("[features] Building train feature table…")
    train = build_feature_table(config.TRAIN_DIR, is_train=True)
    protect = (config.TARGET, config.CASE_ID, "WEEK_NUM", "month_decision",
               "weekday_decision")
    train = prune_columns(train, protect=protect)
    out = config.PROCESSED_DIR / "train_features.parquet"
    train.write_parquet(out)
    print(f"[features] train -> {train.shape} written to {out} "
          f"({time.time() - t0:.0f}s)")
