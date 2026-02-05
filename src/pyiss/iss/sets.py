# src/pyiss/iss/sets.py
import pandas as pd

from ..opus import data_df
from ..constants import X, K, GAP_FACTOR, MIN_GAP_S
from .set_object import ISSSet  # NEW

def infer_set_dfs(seed_opusid: str):
    """
    Low-level engine: returns (neigh_df, set_df).
    Keep this for debugging and for unit tests.
    """
    cols = ["opusid", "time1", "target", "COISSfilter"]

    seed_df = data_df({"opusid": seed_opusid, "limit": 1, "order": "time1,opusid"}, cols)
    seed = seed_df.iloc[0]
    t0 = pd.to_datetime(seed["time1"], utc=True)
    target0 = str(seed["target"])

    base = {"instrument": "Cassini ISS"}

    before = data_df({**base, "time2": seed["time1"], "order": "-time1,opusid", "limit": K}, cols)
    after  = data_df({**base, "time1": seed["time1"], "order":  "time1,opusid", "limit": K}, cols)

    df = pd.concat([before, after], ignore_index=True).drop_duplicates(subset=["opusid"])
    df["time1"] = pd.to_datetime(df["time1"], utc=True)
    df = df.sort_values(["time1", "opusid"]).reset_index(drop=True)

    seed_idx = df.index[df["opusid"] == seed_opusid][0]

    # Δt diagnostics (keep them here if you want; ISSSet can hide them from default user view)
    df["dt_prev_s"] = df["time1"].diff().dt.total_seconds().abs()
    df["dt_next_s"] = df["time1"].shift(-1).sub(df["time1"]).dt.total_seconds().abs()

    dt_vals = df["dt_prev_s"].dropna().values
    dt_sorted = sorted(dt_vals)
    baseline = pd.Series(dt_sorted[: max(1, len(dt_sorted)//2)]).median()
    gap_thresh = max(MIN_GAP_S, GAP_FACTOR * float(baseline))

    L = seed_idx
    while L > 0 and float(df.loc[L, "dt_prev_s"]) <= gap_thresh:
        L -= 1

    R = seed_idx
    while R + 1 < len(df) and float(df.loc[R + 1, "dt_prev_s"]) <= gap_thresh:
        R += 1

    set_df = df.iloc[L:R+1].copy().reset_index(drop=True)

    if len(set_df) > X:
        set_df["abs_dt_seed"] = (set_df["time1"] - t0).dt.total_seconds().abs()
        set_df = set_df.sort_values(["abs_dt_seed", "time1", "opusid"]).head(X)
        set_df = set_df.sort_values(["time1", "opusid"]).reset_index(drop=True)
        set_df = set_df.drop(columns=["abs_dt_seed"])

    return df, set_df

def infer_set(seed_opusid: str) -> ISSSet:
    """
    User-facing API: returns an ISSSet object with nice methods.
    """
    neigh_df, set_df = infer_set_dfs(seed_opusid)
    return ISSSet(set_df=set_df, neigh_df=neigh_df)
