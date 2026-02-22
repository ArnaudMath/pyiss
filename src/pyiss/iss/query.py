from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from ..opus import data_df, to_opus_utc_time
from .display import ISSSetDisplay
from .set_object import ISSSet
from .sets import infer_set

_DEFAULT_QUERY_COLS = ["opusid", "time1", "target", "COISSfilter"]


def _normalize_fetch_cols(cols: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for col in cols:
        c = str(col).strip()
        if not c:
            continue
        if c not in out:
            out.append(c)
    if not out:
        raise ValueError("fetch() received only empty column names.")
    return out


def _to_opus_time(value: object, *, field_name: str) -> str:
    try:
        return to_opus_utc_time(value)
    except Exception as exc:
        raise ValueError(f"Could not parse '{field_name}' time value '{value}'.") from exc


class ISSQueryResult:
    """
    Query result table with helpers to preview and infer sets.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        params: dict[str, object],
        cols: list[str],
    ):
        df = rows.copy()
        if "time1" in df.columns:
            df["time1"] = pd.to_datetime(df["time1"], utc=True, errors="coerce")

        order_cols = [c for c in ["time1", "opusid"] if c in df.columns]
        if order_cols:
            df = df.sort_values(order_cols).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self._df = df
        self._params = dict(params)
        self._cols = list(cols)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def size(self) -> int:
        return len(self._df)

    @property
    def params(self) -> dict[str, object]:
        return dict(self._params)

    @property
    def cols(self) -> list[str]:
        return list(self._cols)

    def show(self, layout: str = "grid") -> ISSSetDisplay:
        if "opusid" not in self._df.columns:
            raise ValueError("show() requires an 'opusid' column in query results.")
        return ISSSetDisplay(self._df.copy(), layout=layout)

    def _resolve_seed_opusid(self, *, which: int = 0, opusid: Optional[str] = None) -> str:
        if "opusid" not in self._df.columns:
            raise ValueError("infer_set() requires an 'opusid' column in query results.")

        if opusid is not None:
            wanted = str(opusid).strip()
            if not wanted:
                raise ValueError("infer_set(opusid=...) received an empty opusid.")
            rows = self._df[self._df["opusid"].astype(str) == wanted]
            if rows.empty:
                raise ValueError(f"OPUSID '{wanted}' is not present in this query result.")
            return str(rows.iloc[0]["opusid"])

        idx = int(which)
        if idx < 0 or idx >= len(self._df):
            raise ValueError(f"which={idx} is out of range for query result size={len(self._df)}.")
        return str(self._df.iloc[idx]["opusid"])

    def infer_set(self, *, which: int = 0, opusid: Optional[str] = None) -> ISSSet:
        seed_opusid = self._resolve_seed_opusid(which=which, opusid=opusid)
        return infer_set(seed_opusid)


class ISSQuery:
    """
    Chainable metadata query builder for Cassini ISS OPUS searches.

    Metadata constraints are expressed via param(key, value), with time(start, end)
    as a convenience helper for time1/time2 windows.
    """

    def __init__(self, *, params: Optional[dict[str, object]] = None):
        base = {"instrument": "Cassini ISS"}
        if params:
            base.update(params)
        self._params = base

    def _spawn(self, *, params: Optional[dict[str, object]] = None) -> "ISSQuery":
        return ISSQuery(params=dict(self._params if params is None else params))

    @property
    def params(self) -> dict[str, object]:
        return dict(self._params)

    def param(self, key: str, value: object) -> "ISSQuery":
        k = str(key).strip()
        if not k:
            raise ValueError("param() key cannot be empty.")
        next_params = dict(self._params)
        next_params[k] = value
        return self._spawn(params=next_params)

    def time(self, start: object, end: object) -> "ISSQuery":
        t1 = _to_opus_time(start, field_name="start")
        t2 = _to_opus_time(end, field_name="end")
        if pd.to_datetime(t1, utc=True) > pd.to_datetime(t2, utc=True):
            raise ValueError("time(start, end) requires start <= end.")
        return self.param("time1", t1).param("time2", t2)

    def limit(self, value: int) -> "ISSQuery":
        n = int(value)
        if n <= 0:
            raise ValueError("limit() requires a strictly positive integer.")
        return self.param("limit", n)

    def fetch(self, *cols: str) -> ISSQueryResult:
        fetch_cols = _normalize_fetch_cols(cols) if cols else list(_DEFAULT_QUERY_COLS)
        try:
            rows = data_df(self._params, fetch_cols)
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{exc}. OPUS rejected this query; verify parameter names/values and requested columns."
            ) from exc
        return ISSQueryResult(rows, params=self._params, cols=fetch_cols)


def query() -> ISSQuery:
    """
    Start a metadata-driven Cassini ISS query.
    """
    return ISSQuery()
