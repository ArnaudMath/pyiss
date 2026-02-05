from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from more_itertools import first
import pandas as pd

from .preview import gallery


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
        seen: list[str] = []
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
        if str(filt).lower() == "all":
            return self._set_df
        return self._set_df[self._set_df[self._filter_col].astype(str) == str(filt)]

    def _one_or_many(self, rows: pd.DataFrame, field: str, *, all: bool):
        vals = rows[field].tolist()
        return vals if all else vals[0]

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
        return rows.reset_index(drop=True) if all else rows.iloc[0]

    # --------------------
    # Display
    # --------------------
    def show(self, filt: str = "all"):
        """
        Show full-preview images for a filter or the entire set.
        """
        rows = self._rows_for_filter(filt)
        gallery(rows)

    # --------------------
    # Diagnostics (opt-in)
    # --------------------
    def diagnostic_window(self) -> pd.DataFrame:
        """
        Return a dataframe containing:
          [row A] + [set rows] + [row Z]

        where:
          - row A = last neighbour before set start (from neigh_df)
          - row Z = first neighbour after set end (from neigh_df)

        dt_prev_s/dt_next_s are computed on THIS window so boundaries are explicit:
          - A.dt_next_s = gap(A -> first set row)
          - first set row dt_prev_s = NA
          - last set row dt_next_s = NA
          - Z.dt_prev_s = gap(last set row -> Z)
        """
        if self._neigh_df is None or len(self._neigh_df) == 0:
            raise ValueError("diagnostic_window() requires neigh_df; create the set via infer_set().")

        set_df = self._set_df.copy()
        set_df["time1"] = pd.to_datetime(set_df["time1"], utc=True)
        set_df = set_df.sort_values(["time1", "opusid"]).reset_index(drop=True)

        neigh = self._neigh_df.copy()
        neigh["time1"] = pd.to_datetime(neigh["time1"], utc=True)
        neigh = neigh.sort_values(["time1", "opusid"]).reset_index(drop=True)

        set_start = set_df["time1"].iloc[0]
        set_end = set_df["time1"].iloc[-1]

        before = neigh[neigh["time1"] < set_start]
        after = neigh[neigh["time1"] > set_end]

        frames = []

        # Row A
        if len(before) > 0:
            A = before.iloc[[-1]].copy()
            A["_role"] = "A"
            frames.append(A)

        # Set rows
        mid = set_df.copy()
        mid["_role"] = "set"
        frames.append(mid)

        # Row Z
        if len(after) > 0:
            Z = after.iloc[[0]].copy()
            Z["_role"] = "Z"
            frames.append(Z)

        win = pd.concat(frames, ignore_index=True)

        # Keep just the useful columns
        keep = [c for c in ["_role", "opusid", "time1", "target", self._filter_col] if c in win.columns]
        win = win[keep].copy()

        # Compute dt on the window
        win["dt_prev_s"] = win["time1"].diff().dt.total_seconds().abs()
        win["dt_next_s"] = win["time1"].shift(-1).sub(win["time1"]).dt.total_seconds().abs()

        set_idxs = win.index[win["_role"] == "set"].tolist()
        if set_idxs:
            first = set_idxs[0]
            last  = set_idxs[-1]

        # If row A exists, mirror A->first gap into first.dt_prev_s
        if first > 0 and win.loc[first - 1, "_role"] == "A":
            win.loc[first, "dt_prev_s"] = win.loc[first - 1, "dt_next_s"]

        # If row Z exists, mirror last->Z gap into last.dt_next_s
        if last + 1 < len(win) and win.loc[last + 1, "_role"] == "Z":
            win.loc[last, "dt_next_s"] = win.loc[last + 1, "dt_prev_s"]

        return win
