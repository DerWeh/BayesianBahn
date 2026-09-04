"""What the published archive calls its columns, and what to do when that changes.

`piebro/deutsche-bahn-data` is upstream. Its schema is not ours to pin, and on
2026-09-03 it replaced `is_canceled` with the pair `arrival_is_canceled` and
`departure_is_canceled` — in *every* monthly file, not only in new ones, so the
nightly rebuild failed on months it had processed successfully for a year. The
failure surfaced as a `ColumnNotFoundError` thrown from inside a lazy query
plan, ten frames deep, naming a column nobody had touched.

Two jobs here.

**Speak both spellings.** Files written before the change and after it are read
by the same code, and `build_recent.py` still writes the old one, so a run
routinely holds a mix. [cancellation] returns the arrival and departure flags
whichever way the file spells them.

**Fail legibly.** [require] checks the columns a reader depends on before the
query is built, so the next rename reports which column went missing from which
file, next to the ones the file does have — rather than surfacing as a plan
that failed at frame ten.

On the semantics of the split: on 400,000 rows of 2026-05 joined between the
old file and the new, `is_canceled` was exactly `arrival_is_canceled OR
departure_is_canceled` — never one alone, never the conjunction. So [either]
reproduces the retired column bit for bit, which is what every caller here
wants, none of them having asked for a distinction that did not exist when they
were written. The split is real information (about 1,200 rows per 400,000 are
cancelled at one end only, a train turned back short of its terminus) and it is
now available to anyone who wants it — `score_events.load_truth` is the one
place where telling a cancelled arrival from a cancelled departure would say
something the caller could use.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

CANCELLED = "is_canceled"
ARRIVAL_CANCELLED = "arrival_is_canceled"
DEPARTURE_CANCELLED = "departure_is_canceled"

# What every reader here needs, under whichever spelling. Named separately from
# any one reader's column list so the check below is about the archive rather
# than about a caller.
CANCELLATION_SPELLINGS = ((CANCELLED,), (ARRIVAL_CANCELLED, DEPARTURE_CANCELLED))


def names_of(source: pl.LazyFrame | pl.DataFrame | Path | str) -> list[str]:
    """The column names of a frame or of a parquet file, without reading it."""
    if isinstance(source, (Path, str)):
        return pl.scan_parquet(source).collect_schema().names()
    if isinstance(source, pl.LazyFrame):
        return source.collect_schema().names()
    return source.columns


def cancellation(source) -> tuple[pl.Expr, pl.Expr]:
    """(arrival, departure) cancellation flags, however the file spells them.

    The old single column stood for both ends, because that is what it meant:
    it was the disjunction, and a reader of it could not have distinguished
    them anyway.
    """
    names = names_of(source)
    if ARRIVAL_CANCELLED in names and DEPARTURE_CANCELLED in names:
        return pl.col(ARRIVAL_CANCELLED), pl.col(DEPARTURE_CANCELLED)
    if CANCELLED in names:
        return pl.col(CANCELLED), pl.col(CANCELLED)
    raise SystemExit(
        f"no cancellation column: expected {CANCELLED!r}, or "
        f"{ARRIVAL_CANCELLED!r} and {DEPARTURE_CANCELLED!r}, among {names}")


def either(source) -> pl.Expr:
    """One flag meaning "this stop did not happen", as `is_canceled` did."""
    arrival, departure = cancellation(source)
    return (arrival | departure).alias(CANCELLED)


def require(source, columns, *, what: str = "the archive") -> list[str]:
    """Check the columns a reader depends on, and say plainly what is missing.

    Called before the query is built. The point is the error message: upstream
    renamed a column once and it cost a nightly run, and the next time that
    happens the report should name the column and the file rather than a frame
    index inside a plan.
    """
    names = names_of(source)
    missing = [c for c in columns if c not in names]
    if missing:
        where = f" in {source}" if isinstance(source, (Path, str)) else ""
        raise SystemExit(
            f"{what} is missing {missing}{where}; it has {names}. Upstream has "
            f"changed its schema before — see pipeline/archive.py.")
    return names
