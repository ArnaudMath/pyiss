from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import warnings

import pandas as pd
import requests

from ..opus import data_df
from .arithmetic import ISSPair
from .display import ISSSetDisplay, normalize_intensity


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()


class AmbiguousSelectionError(ValueError):
    """
    Raised when select() cannot resolve to a single unambiguous observation.
    """


class ISSSet:
    """
    Lightweight wrapper around an inferred ISS filter set.
    """

    def __init__(
        self,
        set_df: pd.DataFrame,
        *,
        neigh_df: Optional[pd.DataFrame] = None,
        selected_image_type: str = "IMG",
        seed_opusid: Optional[str] = None,
        seed_time: Optional[Union[str, pd.Timestamp]] = None,
    ):
        df = set_df.copy()
        if "time1" in df.columns:
            df["time1"] = pd.to_datetime(df["time1"], utc=True)
        df = df.drop(columns=["dt_prev_s", "dt_next_s"], errors="ignore")
        df = df.sort_values(["time1", "opusid"]).reset_index(drop=True)

        self._set_df = df
        self._neigh_df = neigh_df  # optional, for debugging
        self._filter_col = "COISSfilter"
        self._image_type = self._normalize_image_type(selected_image_type)
        self._metadata_cache: dict[str, pd.Series] = {}
        self._seed_opusid = str(seed_opusid) if seed_opusid else None

        st: Optional[pd.Timestamp]
        if seed_time is not None:
            st = pd.to_datetime(seed_time, utc=True)
        elif self._seed_opusid and "opusid" in df.columns and "time1" in df.columns:
            seed_rows = df[df["opusid"].astype(str) == self._seed_opusid]
            st = pd.to_datetime(seed_rows["time1"].iloc[0], utc=True) if not seed_rows.empty else None
        else:
            st = None
        self._seed_time = st

    @staticmethod
    def _normalize_image_type(value: str) -> str:
        image_type = str(value).strip().upper()
        if not image_type:
            raise ValueError("image_type cannot be empty")
        allowed = {"IMG", "CUB", "CAL"}
        if image_type not in allowed:
            warnings.warn(
                f"image_type '{image_type}' is not in the common ISS types {sorted(allowed)}; keeping it as-is.",
                stacklevel=2,
            )
        return image_type

    # --------------------
    # Core data access
    # --------------------
    @property
    def df(self) -> pd.DataFrame:
        """
        The canonical set table (trimmed to core fields).
        """
        core_cols = ["opusid", "time1", "target", self._filter_col]
        existing = [c for c in core_cols if c in self._set_df.columns]
        return self._set_df[existing].copy()

    @property
    def size(self) -> int:
        return len(self._set_df)

    @property
    def available_filters(self) -> list[str]:
        if self._filter_col not in self._set_df.columns:
            return []
        # Keep order of appearance (time-ordered)
        seen: list[str] = []
        for f in self._set_df[self._filter_col].astype(str).tolist():
            if f not in seen:
                seen.append(f)
        return seen

    @property
    def time_window(self) -> TimeWindow:
        if self._set_df.empty:
            raise ValueError("time_window is undefined for an empty ISSSet.")
        return TimeWindow(
            start=self._set_df["time1"].iloc[0],
            end=self._set_df["time1"].iloc[-1],
        )

    @property
    def selected_image_type(self) -> str:
        return self._image_type

    @property
    def filter_counts(self) -> dict[str, int]:
        """
        Per-filter counts in order of first appearance.
        """
        if self._filter_col not in self._set_df.columns:
            return {}
        counts_series = self._set_df[self._filter_col].astype(str).value_counts()
        out: dict[str, int] = {}
        for f in self.available_filters:
            out[f] = int(counts_series.get(f, 0))
        return out

    # --------------------
    # Selection helpers
    # --------------------
    def _rows_for_filter(self, filt: str) -> pd.DataFrame:
        if str(filt).lower() == "all":
            return self._set_df.copy()
        if self._filter_col not in self._set_df.columns:
            return self._set_df.iloc[0:0].copy()
        return self._set_df[self._set_df[self._filter_col].astype(str) == str(filt)].copy()

    def _rows_for_filters(self, filters: tuple[str, ...]) -> pd.DataFrame:
        if not filters:
            return self._set_df.copy()
        wanted = [str(f).strip() for f in filters if str(f).strip()]
        if not wanted or any(f.lower() == "all" for f in wanted):
            return self._set_df.copy()
        if self._filter_col not in self._set_df.columns:
            return self._set_df.iloc[0:0].copy()
        wanted_set = set(wanted)
        return self._set_df[self._set_df[self._filter_col].astype(str).isin(wanted_set)].copy()

    def _spawn(self, rows: pd.DataFrame, *, neigh_df: Optional[pd.DataFrame]) -> "ISSSet":
        return ISSSet(
            set_df=rows,
            neigh_df=neigh_df,
            selected_image_type=self._image_type,
            seed_opusid=self._seed_opusid,
            seed_time=self._seed_time,
        )

    def _fetch_metadata_field(self, field: str) -> pd.Series:
        if field in self._metadata_cache:
            return self._metadata_cache[field]

        if field in self._set_df.columns:
            series = self._set_df[field].reset_index(drop=True)
            self._metadata_cache[field] = series
            return series

        if self._set_df.empty:
            series = pd.Series(dtype="object")
            self._metadata_cache[field] = series
            return series

        start = self._set_df["time1"].iloc[0].isoformat()
        end = self._set_df["time1"].iloc[-1].isoformat()
        params = {
            "instrument": "Cassini ISS",
            "time1": start,
            "time2": end,
            "order": "time1,opusid",
            "limit": max(1000, self.size * 20),
        }
        try:
            fetched = data_df(params, ["opusid", field])
        except requests.HTTPError:
            warnings.warn(
                f"OPUS field '{field}' is unknown or unavailable for this set; returning empty values.",
                stacklevel=2,
            )
            series = pd.Series([pd.NA] * self.size, dtype="object")
            self._metadata_cache[field] = series
            return series
        except Exception as exc:
            warnings.warn(
                f"Could not retrieve OPUS field '{field}' ({exc}); returning empty values.",
                stacklevel=2,
            )
            series = pd.Series([pd.NA] * self.size, dtype="object")
            self._metadata_cache[field] = series
            return series

        if fetched.empty:
            warnings.warn(f"OPUS returned no rows for field '{field}'.", stacklevel=2)
            series = pd.Series([pd.NA] * self.size, dtype="object")
            self._metadata_cache[field] = series
            return series

        lookup = fetched.drop_duplicates(subset=["opusid"], keep="first").set_index("opusid")[field]
        values = self._set_df["opusid"].map(lookup)

        # Fallback path: try direct per-opusid fetch for any unresolved rows.
        missing = values.isna()
        if missing.any():
            for idx, opusid in self._set_df.loc[missing, "opusid"].items():
                try:
                    one = data_df({"opusid": opusid, "limit": 1, "order": "time1,opusid"}, [field])
                except requests.HTTPError:
                    one = pd.DataFrame(columns=[field])
                except Exception:
                    one = pd.DataFrame(columns=[field])
                if not one.empty:
                    values.loc[idx] = one.iloc[0][field]

        series = values.reset_index(drop=True)
        self._metadata_cache[field] = series
        return series

    # --------------------
    # Legacy accessors (v0.1) - intentionally disabled in v0.2+
    # --------------------
    def time1(self, *args, **kwargs):
        raise RuntimeError("time1() is removed in v0.2+. Use set.filter(...).metadata('time1').")

    def target(self, *args, **kwargs):
        raise RuntimeError("target() is removed in v0.2+. Use set.filter(...).metadata('target').")

    def opusid(self, *args, **kwargs):
        raise RuntimeError("opusid() is removed in v0.2+. Use set.filter(...).metadata('opusid').")

    def row(self, *args, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        raise RuntimeError("row() is removed in v0.2+. Use set.filter(...).metadata(...).")

    def metadata(self, *fields: str) -> pd.DataFrame:
        """
        Return arbitrary OPUS metadata columns for the current rows.
        Unknown/unavailable fields are returned as empty values with warnings.
        """
        if not fields:
            raise ValueError("metadata() expects at least one OPUS field name.")

        cols = [str(f).strip() for f in fields]
        if any(not c for c in cols):
            raise ValueError("metadata() field names cannot be empty.")

        out = pd.DataFrame(index=range(self.size))
        for field in cols:
            out[field] = self._fetch_metadata_field(field)
        return out

    # --------------------
    # Core composable operations
    # --------------------
    def filter(self, *filters: str) -> "ISSSet":
        """
        Slice this set by one or more filter names.
        """
        if not filters:
            raise ValueError("filter() expects at least one filter name.")

        wanted = [str(f).strip() for f in filters if str(f).strip()]
        if not wanted:
            raise ValueError("filter() received only empty filter names.")

        if any(f.lower() == "all" for f in wanted):
            rows = self._set_df.copy()
        elif self._filter_col not in self._set_df.columns:
            rows = self._set_df.iloc[0:0].copy()
        else:
            wanted_set = set(wanted)
            rows = self._set_df[self._set_df[self._filter_col].astype(str).isin(wanted_set)].copy()

        rows = rows.sort_values(["time1", "opusid"]).reset_index(drop=True)
        return self._spawn(rows, neigh_df=None)

    def image_type(self, value: str) -> "ISSSet":
        """
        Select the desired downstream image type (e.g. IMG/CUB/CAL) without plotting.
        """
        selected = self._normalize_image_type(value)
        return ISSSet(
            set_df=self._set_df.copy(),
            neigh_df=self._neigh_df,
            selected_image_type=selected,
            seed_opusid=self._seed_opusid,
            seed_time=self._seed_time,
        )

    def select(
        self,
        filter_name: Optional[str] = None,
        *,
        which: Optional[int] = None,
        nearest: Optional[str] = None,
        t: Optional[Union[str, pd.Timestamp]] = None,
        opusid: Optional[str] = None,
    ) -> "ISSSet":
        """
        Resolve exactly one observation from this set.

        Examples:
            s.select("RED", which=0)
            s.select("RED", nearest="seed")
            s.select("RED", nearest="time", t="2015-10-14T10:10:00Z")
            s.select(opusid="co-iss-n123...")
        """
        if opusid is not None:
            if any(v is not None for v in (filter_name, which, nearest, t)):
                raise ValueError(
                    "select(opusid=...) is exclusive; do not also pass filter_name/which/nearest/t."
                )
            rows = self._set_df[self._set_df["opusid"].astype(str) == str(opusid)].copy()
            if rows.empty:
                raise ValueError(f"No observation with opusid '{opusid}' in this set.")
            if len(rows) > 1:
                raise AmbiguousSelectionError(
                    f"Multiple rows found for opusid '{opusid}'. This set should contain unique opusid values."
                )
            return self._spawn(rows.reset_index(drop=True), neigh_df=None)

        if filter_name is None:
            raise ValueError("select() requires either filter_name or opusid.")

        filt = str(filter_name).strip()
        if not filt:
            raise ValueError("select() received an empty filter_name.")

        rows = self._rows_for_filter(filt).sort_values(["time1", "opusid"]).reset_index(drop=True)
        n = len(rows)
        if n == 0:
            raise ValueError(f"Filter '{filt}' is not present in this set.")

        if which is not None and nearest is not None:
            raise ValueError("select() accepts either 'which' or 'nearest', not both.")
        if t is not None and str(nearest).lower() != "time":
            raise ValueError("Parameter 't' is only valid with nearest='time'.")

        if which is not None:
            idx = int(which)
            if idx < 0 or idx >= n:
                raise ValueError(f"Index out of range for filter '{filt}': which={idx}, size={n}.")
            return self._spawn(rows.iloc[[idx]].reset_index(drop=True), neigh_df=None)

        if nearest is not None:
            mode = str(nearest).strip().lower()
            if mode == "seed":
                if self._seed_time is None:
                    raise ValueError(
                        "nearest='seed' is unavailable because seed time is not stored on this set."
                    )
                target_time = self._seed_time
            elif mode == "time":
                if t is None:
                    raise ValueError("nearest='time' requires 't' (ISO string or pandas Timestamp).")
                try:
                    target_time = pd.to_datetime(t, utc=True)
                except Exception as exc:
                    raise ValueError(f"Could not parse time value '{t}'.") from exc
            else:
                raise ValueError("nearest must be one of: 'seed', 'time'.")

            dt = (rows["time1"] - target_time).abs().dt.total_seconds()
            idx = int(dt.idxmin())
            return self._spawn(rows.loc[[idx]].reset_index(drop=True), neigh_df=None)

        if n == 1:
            return self._spawn(rows.iloc[[0]].reset_index(drop=True), neigh_df=None)

        max_idx = n - 1
        raise AmbiguousSelectionError(
            "\n".join(
                [
                    f"Filter '{filt}' occurs {n} times in this set.",
                    "Use one of:",
                    f"  - s.select('{filt}', which=0..{max_idx})",
                    f"  - s.select('{filt}', nearest='seed')",
                    "  - s.select(opusid='...')",
                ]
            )
        )

    # --------------------
    # Display
    # --------------------
    def show(self, *filters: str, layout: str = "grid") -> ISSSetDisplay:
        """
        Show preview images for one or more filters (or the whole set).
        Returns a chainable display object:
            set.show("GRN", "UV3", layout="row").image_size("medium").image_calibrated(True)
        """
        rows = self._rows_for_filters(filters)
        return ISSSetDisplay(rows, layout=layout)

    # --------------------
    # Scientific operations (v0.3)
    # --------------------
    def pair(
        self,
        filter_a: Union[str, "ISSSet"],
        filter_b: Union[str, "ISSSet"],
        *,
        max_dt_s: Optional[float] = None,
        image_size: str = "medium",
        image_calibrated: Optional[bool] = None,
        intensity: str = "DN",
    ) -> ISSPair:
        """
        Build a strict pair object for arithmetic between two filters or selected observations.

        Accepted operands:
          - filter string: e.g. "IR1"
          - single-observation ISSSet: e.g. obs0 = set.select("IR1", which=0)

        Temporal behavior:
          - by default (max_dt_s=None): nearest-time matching is used without rejection
          - if max_dt_s is provided: reject pairs whose time mismatch exceeds this bound
        """
        if max_dt_s is not None and max_dt_s <= 0:
            raise ValueError("max_dt_s must be > 0 when provided.")

        def _resolve_operand(operand: Union[str, "ISSSet"], side: str) -> tuple[pd.DataFrame, str, bool]:
            # Returns: (rows, filter_name, is_single_selected)
            rows_from_obj: Optional[pd.DataFrame] = None

            # Fast path for same-class ISSSet instances.
            if isinstance(operand, ISSSet):
                if operand.size != 1:
                    raise AmbiguousSelectionError(
                        f"pair() operand '{side}' must contain exactly one row when passing an ISSSet. "
                        "Use select(...) first."
                    )
                rows_from_obj = operand.df.copy()
            else:
                # Notebook-safe path: accept ISSSet-like objects across module reloads.
                # This avoids brittle isinstance failures when class identity changes.
                if hasattr(operand, "df") and hasattr(operand, "size"):
                    try:
                        maybe_size = int(getattr(operand, "size"))
                        maybe_df = getattr(operand, "df")
                        if maybe_size == 1 and isinstance(maybe_df, pd.DataFrame):
                            rows_from_obj = maybe_df.copy()
                    except Exception:
                        rows_from_obj = None

            if rows_from_obj is not None:
                if self._filter_col not in rows_from_obj.columns:
                    raise ValueError(
                        f"pair() operand '{side}' does not expose '{self._filter_col}'."
                    )
                filt = str(rows_from_obj[self._filter_col].iloc[0]).strip()
                if not filt:
                    raise ValueError(f"pair() operand '{side}' has an empty filter name.")
                rows = rows_from_obj.sort_values(["time1", "opusid"]).reset_index(drop=True)
                return rows, filt, True

            filt = str(operand).strip()
            if not filt:
                raise ValueError(f"pair() operand '{side}' must be a non-empty filter name or selected ISSSet.")
            rows = self._rows_for_filter(filt).sort_values(["time1", "opusid"]).reset_index(drop=True)
            return rows, filt, False

        left, fa, left_single = _resolve_operand(filter_a, "filter_a")
        right, fb, right_single = _resolve_operand(filter_b, "filter_b")

        if left.empty or right.empty:
            raise ValueError(f"pair() could not find rows for filters '{fa}' and '{fb}'.")

        if left_single and not right_single:
            t_left = left["time1"].iloc[0]
            dt_candidates = (right["time1"] - t_left).abs().dt.total_seconds()
            best_right_idx = int(dt_candidates.idxmin())
            best_dt = float(dt_candidates.loc[best_right_idx])
            if max_dt_s is not None and best_dt > float(max_dt_s):
                raise ValueError(
                    f"pair() could not find a '{fb}' match within {max_dt_s}s for selected '{fa}'. "
                    f"Nearest dt={best_dt:.3f}s."
                )
            right = right.loc[[best_right_idx]].reset_index(drop=True)

        if right_single and not left_single:
            t_right = right["time1"].iloc[0]
            dt_candidates = (left["time1"] - t_right).abs().dt.total_seconds()
            best_left_idx = int(dt_candidates.idxmin())
            best_dt = float(dt_candidates.loc[best_left_idx])
            if max_dt_s is not None and best_dt > float(max_dt_s):
                raise ValueError(
                    f"pair() could not find a '{fa}' match within {max_dt_s}s for selected '{fb}'. "
                    f"Nearest dt={best_dt:.3f}s."
                )
            left = left.loc[[best_left_idx]].reset_index(drop=True)

        if len(left) != len(right):
            raise ValueError(
                f"pair() requires equal row counts unless one side is pre-selected; got {fa}={len(left)} and {fb}={len(right)}."
            )

        dt_s = (left["time1"] - right["time1"]).abs().dt.total_seconds()
        worst_dt = float(dt_s.max())
        if max_dt_s is not None and worst_dt > float(max_dt_s):
            raise ValueError(
                f"pair() rejected due to time mismatch: max dt={worst_dt:.3f}s > max_dt_s={max_dt_s}."
            )

        if "target" in left.columns and "target" in right.columns:
            same_target = (left["target"].astype(str).values == right["target"].astype(str).values).all()
            if not same_target:
                raise ValueError("pair() rejected: target mismatch between filter rows.")

        calibrated = (self._image_type == "CAL") if image_calibrated is None else bool(image_calibrated)
        norm_intensity = normalize_intensity(intensity)
        if norm_intensity != "DN" and not calibrated:
            raise ValueError("pair() with non-DN intensity requires image_calibrated=True.")

        pair_df = pd.DataFrame(
            {
                "left_opusid": left["opusid"].astype(str).tolist(),
                "right_opusid": right["opusid"].astype(str).tolist(),
                "left_time1": left["time1"].tolist(),
                "right_time1": right["time1"].tolist(),
                "target": left["target"].astype(str).tolist() if "target" in left.columns else [""] * len(left),
                "left_filter": [fa] * len(left),
                "right_filter": [fb] * len(left),
                "dt_s": dt_s.astype(float).tolist(),
            }
        )

        return ISSPair(
            pair_df,
            filter_a=fa,
            filter_b=fb,
            image_size=image_size,
            image_calibrated=calibrated,
            intensity=norm_intensity,
            image_type=self._image_type,
        )

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
        if set_df.empty:
            raise ValueError("diagnostic_window() is undefined for an empty ISSSet.")
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
        if not set_idxs:
            return win
        first = set_idxs[0]
        last = set_idxs[-1]

        # If row A exists, mirror A->first gap into first.dt_prev_s
        if first > 0 and win.loc[first - 1, "_role"] == "A":
            win.loc[first, "dt_prev_s"] = win.loc[first - 1, "dt_next_s"]

        # If row Z exists, mirror last->Z gap into last.dt_next_s
        if last + 1 < len(win) and win.loc[last + 1, "_role"] == "Z":
            win.loc[last, "dt_next_s"] = win.loc[last + 1, "dt_prev_s"]

        return win
