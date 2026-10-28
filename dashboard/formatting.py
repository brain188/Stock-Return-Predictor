"""How numbers are written: one place, so every page formats a Sharpe the same way.

Missing values are written as an em dash and never as zero: a dash says "not
recorded", a zero would be a claim.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

MISSING = "\u2014"
# A true minus sign reads better in a monospaced column than a hyphen.
MINUS = "\u2212"


def is_missing(value: Any) -> bool:
    """True for None, NaN, NaT and non-finite floats."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) or (isinstance(value, float) and not math.isfinite(value))
    except (TypeError, ValueError):
        return False


def number(value: Any, digits: int = 2, signed: bool = False) -> str:
    """A plain number with fixed digits."""
    if is_missing(value):
        return MISSING
    text = f"{float(value):+,.{digits}f}" if signed else f"{float(value):,.{digits}f}"
    return text.replace("-", MINUS)


def pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    """A fraction as a percentage: 0.1234 -> 12.3%."""
    if is_missing(value):
        return MISSING
    return number(float(value) * 100, digits, signed) + "%"


def pp(value: Any, digits: int = 2) -> str:
    """A difference of two fractions in percentage points, always signed."""
    if is_missing(value):
        return MISSING
    return number(float(value) * 100, digits, signed=True) + " pp"


def bps(value: Any, digits: int = 1, signed: bool = False) -> str:
    """A fraction in basis points: 0.0005 -> 5.0 bps."""
    if is_missing(value):
        return MISSING
    return number(float(value) * 10_000, digits, signed) + " bps"


def sharpe(value: Any, signed: bool = False) -> str:
    """A Sharpe ratio, two decimals."""
    return number(value, 2, signed)


def integer(value: Any) -> str:
    """A count with thousands separators."""
    if is_missing(value):
        return MISSING
    return f"{round(float(value)):,}"


def date(value: Any) -> str:
    """An ISO date, from a timestamp or a string."""
    if is_missing(value):
        return MISSING
    return str(pd.Timestamp(value).date())


def date_range(start: Any, end: Any) -> str:
    """Two dates joined by an arrow, as the design writes ranges."""
    return f"{date(start)} → {date(end)}"


def tone(value: Any, *, good_when: str = "positive", neutral_band: float = 0.0) -> str:
    """Which semantic colour a value earns: "pos", "neg" or "neutral".

    Colour is never the only signal: the sign and the label carry the meaning,
    and this only decides the accent.
    """
    if is_missing(value):
        return "neutral"
    x = float(value)
    if abs(x) <= neutral_band:
        return "neutral"
    positive = x > 0
    if good_when == "negative":
        positive = not positive
    return "pos" if positive else "neg"
