"""Disproportionality statistics on a 2x2 table. Pure functions, no I/O.

            reaction   other reactions
drug           a            b
other drugs    c            d
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Table:
    a: int
    b: int
    c: int
    d: int

    def __post_init__(self):
        if min(self.a, self.b, self.c, self.d) < 0:
            raise ValueError("table cells cannot be negative")

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d


@dataclass(frozen=True)
class Thresholds:
    """Evans screening rule."""

    prr_min: float = 2.0
    chi2_min: float = 4.0
    a_min: int = 3


@dataclass(frozen=True)
class Signal:
    table: Table
    prr: Optional[float]
    chi2: Optional[float]
    flagged: bool


def prr(t: Table) -> Optional[float]:
    """[a/(a+b)] / [c/(c+d)]. None when undefined (empty row, or no other-drug reports of the reaction)."""
    if t.a + t.b == 0 or t.c + t.d == 0 or t.c == 0:
        return None
    return (t.a / (t.a + t.b)) / (t.c / (t.c + t.d))


def chi_square(t: Table) -> Optional[float]:
    """Pearson chi-square, 1 degree of freedom, no continuity correction. None when a margin is zero."""
    row1, row2 = t.a + t.b, t.c + t.d
    col1, col2 = t.a + t.c, t.b + t.d
    if 0 in (row1, row2, col1, col2):
        return None
    return t.n * (t.a * t.d - t.b * t.c) ** 2 / (row1 * row2 * col1 * col2)


def evaluate(t: Table, th: Thresholds = Thresholds()) -> Signal:
    p, x = prr(t), chi_square(t)
    flagged = p is not None and x is not None and p >= th.prr_min and x >= th.chi2_min and t.a >= th.a_min
    return Signal(table=t, prr=p, chi2=x, flagged=flagged)
