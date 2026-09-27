"""TSV I/O helpers.

Two rules matter for this challenge and are easy to get wrong on Windows:

1. Files must be read/written as UTF-8 (business names contain Devanagari,
   Bengali, Kannada, ... script) with ``sep="\\t"`` and no NA-sniffing (an
   empty address must stay ``""``, not become ``NaN``, and a literal address
   token like ``"null"`` must NOT be read back as a NaN).
2. The official validator (``utils/validate_submission.py``) splits input on
   ``"\\n"`` only. Pandas' default line terminator on Windows is ``"\\r\\n"``,
   which corrupts every ID in the last column. We always write with
   ``lineterminator="\\n"``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

TSV_READ_KW = dict(
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
    quoting=csv.QUOTE_NONE,
    encoding="utf-8",
)


def read_tsv(path: str | Path, usecols: Iterable[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, nrows=nrows, **TSV_READ_KW)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )


def read_source(paths, split: str, source: str) -> pd.DataFrame:
    """Read one raw source file (source1/source2/source3) for a split."""
    p = paths.raw(source)
    df = read_tsv(p)
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    return df


def read_ground_truth(paths) -> pd.DataFrame:
    return read_tsv(paths.ground_truth())
