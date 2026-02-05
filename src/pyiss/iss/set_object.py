from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd

from .preview import gallery  # your existing HTML preview
from ..opus import full_preview_url


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


class ISSSet:
    """
    Lightweight wrapper around an inferred ISS filter set.
    """

    def __init__(self, set_df: pd.DataFrame, *, neigh_df: Optional[pd.DataFrame] = None):
        df = set_df.copy()
        df["time1"] = pd.to_datetime(df["time1"], utc=True)
        df = df.drop(columns=["dt_prev_s", "dt_next_s"], errors="ignore")
        df = df.sort_values(["time1", "opusid"]).reset_index(drop=True)

        self._set_df = df
        self._neigh_df = neigh_df  # optional, for debugging
        self._filter_col = "COISSfilter"

    # --------------------
    # Core data access
    # --------------------
    @property
    def df(self) -> pd.DataFrame:
        """The canonical set table."""
        return self._set_df

    @property
    def size(self) -> int:
        return len(self._set_df)

    @property
    def available_filters(self) -> list[str]:
        # Keep order of appearance (time-ordered)
        seen = []
        for f in self._set_df[self._filter_col].astype(str).tolist():
            if f not in seen:
                seen.append(f)
        return seen

    @property
    def time_window(self) -> TimeWindow:
        return TimeWindow(
            start=self._set_df["time1"].iloc[0],
            end=self._set_df["time1"].iloc[-1],
        )

    # --------------------
    # Selection helpers
    # --------------------
    def _rows_for_filter(self, filt: str) -> pd.DataFrame:
        if filt == "all":
            return self._set_df
        return self._set_df[self._set_df[self._filter_col].astype(str) == str(filt)]

    def _one_or_many(self, rows: pd.DataFrame, field: str, *, all: bool):
        vals = rows[field].tolist()
        if all:
            return vals
        return vals[0]  # assumes at least 1 match

    # --------------------
    # Metadata getters
    # --------------------
    def time1(self, filt: str, *, all: bool = False):
        rows = self._rows_for_filter(filt)
        return self._one_or_many(rows, "time1", all=all)

    def target(self, filt: str, *, all: bool = False):
        rows = self._rows_for_filter(filt)
        return self._one_or_many(rows, "target", all=all)

    def opusid(self, filt: str, *, all: bool = False):
        rows = self._rows_for_filter(filt)
        return self._one_or_many(rows, "opusid", all=all)

    def row(self, filt: str, *, all: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """Return the row(s) for a filter; useful for quick inspection."""
        rows = self._rows_for_filter(filt)
        if all:
            return rows.reset_index(drop=True)
        return rows.iloc[0]

    # --------------------
    # Display
    # --------------------
    def show(self, filt: str = "all"):
        """
        Show full-preview images for a filter or the entire set.
        """
        rows = self._rows_for_filter(filt)
        # Reuse your gallery function which expects a df with opusid/time1/COISSfilter
        gallery(rows)

    # --------------------
    # Diagnostics (opt-in)
    # --------------------
    def diagnostics(self) -> pd.DataFrame:
        """
        Returns a copy of the set df with:
          - dt_prev_s / dt_next_s: within-set gaps
          - boundary gaps folded into dt_prev_s (first row) and dt_next_s (last row)
        Requires neigh_df to be present (it is if created via infer_set()).
        """
        df = self._set_df.copy()
        df = df.sort_values(["time1", "opusid"]).reset_index(drop=True)

        df["dt_prev_s"] = df["time1"].diff().dt.total_seconds().abs()
        df["dt_next_s"] = df["time1"].shift(-1).sub(df["time1"]).dt.total_seconds().abs()

        if self._neigh_df is not None and len(self._neigh_df) > 0:
            neigh = self._neigh_df.copy()
            neigh["time1"] = pd.to_datetime(neigh["time1"], utc=True)
            neigh = neigh.sort_values(["time1", "opusid"]).reset_index(drop=True)

            set_start = df["time1"].iloc[0]
            set_end = df["time1"].iloc[-1]

            before = neigh[neigh["time1"] < set_start]
            after = neigh[neigh["time1"] > set_end]

            if len(before) > 0:
                dt_before = (set_start - before["time1"].iloc[-1]).total_seconds()
                df.loc[0, "dt_prev_s"] = float(dt_before)

            if len(after) > 0:
                dt_after = (after["time1"].iloc[0] - set_end).total_seconds()
                df.loc[len(df) - 1, "dt_next_s"] = float(dt_after)

        return df
