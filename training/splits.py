"""Frozen dataset splits + walk-forward fold definitions.

Every model in the comparison must train and evaluate on the SAME splits.
This module is the single source of truth for those splits.

Two split modes:
  - "fixed": one train / val / test split with anchored dates. The default
    for the main comparison; ensures all models see identical data.
  - "walkforward": one of named historical crisis folds (GFC, COVID, Bear-2022,
    Eurozone-2011) — train on everything before the crisis, test on the crisis
    window. Used in Stage 4 for temporal-robustness eval.

Dates are stored as ISO strings; convert via pandas at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    description: str = ""


# ---------- Fixed primary split (used for the main comparison) ----------
#
# Train: 1995-01-01 -> 2015-12-31 — covers Asian crisis '97, dotcom, GFC, Eurozone.
# Val:   2016-01-01 -> 2018-12-31 — relatively calm; lets us pick HPs without
#        leaking late-cycle stress.
# Test:  2019-01-01 -> 2024-12-31 — COVID crash, 2022 bear, 2023 banking
#        mini-crisis. Held out for final reporting only.

PRIMARY_SPLIT_B = TemporalSplit(
    name="primary_b",
    train_start="1995-01-01",
    train_end="2015-12-31",
    val_start="2016-01-01",
    val_end="2018-12-31",
    test_start="2019-01-01",
    test_end="2024-12-31",
    description="Task B (market) primary split: trains through Eurozone, "
                "validates on calm '16-'18, tests on COVID + 2022 bear + 2023 banking.",
)

# Task A (macro / Laeven & Valencia) — long-history split.
# Crises are sparser in macro data, so we use a longer train window.
PRIMARY_SPLIT_A = TemporalSplit(
    name="primary_a",
    train_start="1870-01-01",
    train_end="1999-12-31",
    val_start="2000-01-01",
    val_end="2006-12-31",
    test_start="2007-01-01",
    test_end="2017-12-31",
    description="Task A (macro) primary split: trains through 130y of crises, "
                "validates pre-GFC, tests on GFC + Eurozone in the test window.",
)


def get_primary_split(task: str) -> TemporalSplit:
    task = task.upper()
    if task == "A":
        return PRIMARY_SPLIT_A
    if task == "B":
        return PRIMARY_SPLIT_B
    raise ValueError(f"Unknown task '{task}'. Expected 'A' or 'B'.")


# ---------- Walk-forward folds (Stage 4) ----------
#
# Each fold trains on everything before the crisis window and tests inside it.
# Validation is the last 12 months before test start so HP selection doesn't
# peek at the crisis.

WALKFORWARD_FOLDS: Dict[str, TemporalSplit] = {
    "gfc": TemporalSplit(
        name="gfc",
        train_start="1995-01-01",
        train_end="2006-12-31",
        val_start="2007-01-01",
        val_end="2007-12-31",
        test_start="2008-01-01",
        test_end="2009-12-31",
        description="Global Financial Crisis fold.",
    ),
    "eurozone2011": TemporalSplit(
        name="eurozone2011",
        train_start="1995-01-01",
        train_end="2009-12-31",
        val_start="2010-01-01",
        val_end="2010-12-31",
        test_start="2011-01-01",
        test_end="2012-12-31",
        description="Eurozone sovereign debt crisis fold.",
    ),
    "covid": TemporalSplit(
        name="covid",
        train_start="1995-01-01",
        train_end="2018-12-31",
        val_start="2019-01-01",
        val_end="2019-12-31",
        test_start="2020-01-01",
        test_end="2020-12-31",
        description="COVID crash fold.",
    ),
    "bear2022": TemporalSplit(
        name="bear2022",
        train_start="1995-01-01",
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        description="2022 bear market fold.",
    ),
}


def get_walkforward_fold(name: str) -> TemporalSplit:
    name = name.lower()
    if name not in WALKFORWARD_FOLDS:
        raise ValueError(
            f"Unknown walk-forward fold '{name}'. Known: {sorted(WALKFORWARD_FOLDS)}"
        )
    return WALKFORWARD_FOLDS[name]


def split_dates(split: TemporalSplit) -> Tuple[str, str, str, str]:
    """Return (train_end, val_end) which is what `PanelCrisisDataset` accepts."""
    return split.train_end, split.val_end
