# src/pyiss/iss/sets.py
import math
import requests
import pandas as pd

from ..opus import data_df
from ..constants import X, K, GAP_FACTOR, MIN_GAP_S
from .set_object import ISSSet  # NEW


def _norm_target_key(target: str) -> str:
    return "".join(ch for ch in str(target).upper() if ch.isalnum())


def _surfacegeo_cols(target: str) -> list[str]:
    t = _norm_target_key(target)
    if not t:
        return []
    return [
        f"SURFACEGEO{t}_planetographiclatitude1",
        f"SURFACEGEO{t}_planetographiclatitude2",
        f"SURFACEGEO{t}_planetocentriclatitude1",
        f"SURFACEGEO{t}_planetocentriclatitude2",
        f"SURFACEGEO{t}_IAUwestlongitude1",
        f"SURFACEGEO{t}_IAUwestlongitude2",
        f"SURFACEGEO{t}_IAUeastlongitude1",
        f"SURFACEGEO{t}_IAUeastlongitude2",
    ]


def _center_from_range(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    return 0.5 * (a_num + b_num)


def _center_lon_from_range(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    diff = (b_num - a_num).abs()
    wrap = diff > 180.0
    b2 = b_num.where(~wrap, b_num + 360.0)
    mid = 0.5 * (a_num + b2)
    return mid.mod(360.0)


def _geo_prev_deg(df: pd.DataFrame, *, target: str) -> pd.Series:
    t = _norm_target_key(target)
    if not t:
        return pd.Series([float("nan")] * len(df), index=df.index)

    lat1 = f"SURFACEGEO{t}_planetographiclatitude1"
    lat2 = f"SURFACEGEO{t}_planetographiclatitude2"
    if lat1 in df.columns and lat2 in df.columns:
        latc = _center_from_range(df[lat1], df[lat2])
    else:
        lat1 = f"SURFACEGEO{t}_planetocentriclatitude1"
        lat2 = f"SURFACEGEO{t}_planetocentriclatitude2"
        if lat1 in df.columns and lat2 in df.columns:
            latc = _center_from_range(df[lat1], df[lat2])
        else:
            return pd.Series([float("nan")] * len(df), index=df.index)

    lon1 = f"SURFACEGEO{t}_IAUwestlongitude1"
    lon2 = f"SURFACEGEO{t}_IAUwestlongitude2"
    if lon1 in df.columns and lon2 in df.columns:
        lonc = _center_lon_from_range(df[lon1], df[lon2])
    else:
        lon1 = f"SURFACEGEO{t}_IAUeastlongitude1"
        lon2 = f"SURFACEGEO{t}_IAUeastlongitude2"
        if lon1 in df.columns and lon2 in df.columns:
            lonc = _center_lon_from_range(df[lon1], df[lon2])
        else:
            return pd.Series([float("nan")] * len(df), index=df.index)

    dlat = latc.diff()
    dlon = ((lonc.diff() + 180.0) % 360.0) - 180.0
    scale = latc.map(lambda v: math.cos(math.radians(v)) if pd.notna(v) else float("nan"))
    return (dlat * dlat + (dlon * scale) * (dlon * scale)).pow(0.5)


def _adaptive_gap_threshold(dt_vals: list[float], *, gap_factor: float) -> tuple[float, bool]:
    dt = pd.Series(dt_vals, dtype="float64")
    dt = dt[dt.notna() & (dt > 0)]
    if len(dt) == 0:
        return float("inf"), False

    # Trim only the top 2% edge gaps so local cadence still dominates.
    dt = dt.sort_values().reset_index(drop=True)
    if len(dt) >= 10:
        trim_n = max(1, int(0.02 * len(dt)))
        core = dt.iloc[: len(dt) - trim_n]
    else:
        core = dt
    if len(core) < 4:
        core = dt

    # Baseline fallback from lower half (current behavior family).
    dt_sorted = core.sort_values().tolist()
    lo_half = dt_sorted[: max(1, len(dt_sorted) // 2)]
    baseline = float(pd.Series(lo_half, dtype="float64").median())
    fallback = max(MIN_GAP_S, gap_factor * baseline)

    # 1D two-means in log-space: detect two local cadence scales.
    x = core.map(math.log).tolist()
    x_sorted = sorted(x)
    c1 = x_sorted[max(0, int(0.30 * (len(x_sorted) - 1)))]
    c2 = x_sorted[max(0, int(0.70 * (len(x_sorted) - 1)))]

    for _ in range(20):
        g1 = [v for v in x if abs(v - c1) <= abs(v - c2)]
        g2 = [v for v in x if abs(v - c2) < abs(v - c1)]
        if not g1 or not g2:
            break
        nc1 = sum(g1) / len(g1)
        nc2 = sum(g2) / len(g2)
        if abs(nc1 - c1) + abs(nc2 - c2) < 1e-6:
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2

    # Recompute final assignments and require clear two-scale structure.
    labs = [0 if abs(v - c1) <= abs(v - c2) else 1 for v in x]
    g1 = [v for v, lab in zip(x, labs) if lab == 0]
    g2 = [v for v, lab in zip(x, labs) if lab == 1]
    if not g1 or not g2:
        return fallback, False

    m1 = sum(g1) / len(g1)
    m2 = sum(g2) / len(g2)
    lo, hi = (m1, m2) if m1 <= m2 else (m2, m1)
    ratio = math.exp(hi - lo)

    n = len(x)
    min_frac = min(len(g1), len(g2)) / float(n)
    mid = 0.5 * (lo + hi)
    band = max(0.05, 0.20 * (hi - lo))
    near_mid_frac = sum(1 for v in x if abs(v - mid) <= band) / float(n)
    near_mid_limit = 0.35 if n < 16 else 0.25

    # Accept bimodality only when separation is clear and the middle is sparse.
    if ratio >= 1.45 and min_frac >= 0.15 and near_mid_frac <= near_mid_limit:
        return math.exp(mid), True
    return fallback, False


def _adaptive_geo_threshold(geo_vals: list[float]) -> float:
    geo = pd.Series(geo_vals, dtype="float64")
    geo = geo[geo.notna() & (geo > 0)]
    if len(geo) < 8:
        return float("inf")
    thr, bimodal = _adaptive_gap_threshold(geo.tolist(), gap_factor=2.0)
    return thr if bimodal else float("inf")


def _fetch_data_with_optional_geo(params: dict, base_cols: list[str], geo_cols: list[str]) -> pd.DataFrame:
    cols_ext = base_cols + geo_cols
    try:
        return data_df(params, cols_ext)
    except requests.HTTPError:
        # Some targets may not expose every SURFACEGEO column; keep time-only path working.
        return data_df(params, base_cols)


def _time_bounds_for_thresh(df: pd.DataFrame, *, seed_idx: int, gap_thresh: float) -> tuple[int, int]:
    L = seed_idx
    while L > 0 and float(df.loc[L, "dt_prev_s"]) <= gap_thresh:
        L -= 1

    R = seed_idx
    while R + 1 < len(df) and float(df.loc[R + 1, "dt_prev_s"]) <= gap_thresh:
        R += 1

    return L, R


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

    geo_cols = _surfacegeo_cols(target0)
    before = _fetch_data_with_optional_geo(
        {**base, "time2": seed["time1"], "order": "-time1,opusid", "limit": K},
        cols,
        geo_cols,
    )
    after = _fetch_data_with_optional_geo(
        {**base, "time1": seed["time1"], "order": "time1,opusid", "limit": K},
        cols,
        geo_cols,
    )

    df = pd.concat([before, after], ignore_index=True).drop_duplicates(subset=["opusid"])
    df["time1"] = pd.to_datetime(df["time1"], utc=True)
    df = df.sort_values(["time1", "opusid"]).reset_index(drop=True)

    seed_idx = df.index[df["opusid"] == seed_opusid][0]

    # Δt diagnostics (keep them here if you want; ISSSet can hide them from default user view)
    df["dt_prev_s"] = df["time1"].diff().dt.total_seconds().abs()
    df["dt_next_s"] = df["time1"].shift(-1).sub(df["time1"]).dt.total_seconds().abs()

    dt_series = df["dt_prev_s"].dropna().astype(float)
    gap_thresh, time_bimodal = _adaptive_gap_threshold(
        dt_series.tolist(), gap_factor=GAP_FACTOR
    )

    # Legacy dynamic threshold retained as rescue when adaptive threshold collapses to singleton.
    dt_sorted = sorted(dt_series.tolist())
    lo_half = dt_sorted[: max(1, len(dt_sorted) // 2)] if dt_sorted else [float("inf")]
    fallback_gap = max(MIN_GAP_S, GAP_FACTOR * float(pd.Series(lo_half).median()))

    # Primary set bounds: time-only.
    time_L, time_R = _time_bounds_for_thresh(df, seed_idx=seed_idx, gap_thresh=gap_thresh)
    if (time_R - time_L + 1) == 1 and gap_thresh < fallback_gap:
        time_L, time_R = _time_bounds_for_thresh(df, seed_idx=seed_idx, gap_thresh=fallback_gap)
        gap_thresh = fallback_gap
        time_bimodal = False

    # Secondary refinement: optional geometry tie-breaker inside time bounds.
    L, R = time_L, time_R
    df["geo_prev_deg"] = _geo_prev_deg(df, target=target0)
    geo_thresh = _adaptive_geo_threshold(df["geo_prev_deg"].dropna().tolist())
    near_time_boundary = 0.85 * gap_thresh

    if time_bimodal and math.isfinite(geo_thresh):
        L = seed_idx
        while L > time_L:
            dt_prev = float(df.loc[L, "dt_prev_s"])
            # Geometry can only split when time gap is near the boundary.
            if dt_prev >= near_time_boundary:
                geo_prev = (
                    float(df.loc[L, "geo_prev_deg"])
                    if pd.notna(df.loc[L, "geo_prev_deg"])
                    else float("nan")
                )
                if math.isfinite(geo_prev) and geo_prev > geo_thresh:
                    break
            L -= 1

        R = seed_idx
        while R < time_R:
            dt_prev = float(df.loc[R + 1, "dt_prev_s"])
            if dt_prev >= near_time_boundary:
                geo_prev = (
                    float(df.loc[R + 1, "geo_prev_deg"])
                    if pd.notna(df.loc[R + 1, "geo_prev_deg"])
                    else float("nan")
                )
                if math.isfinite(geo_prev) and geo_prev > geo_thresh:
                    break
            R += 1

        # Guardrail: geometry should not collapse a valid time-set to singleton.
        if (R - L + 1) == 1 and (time_R - time_L + 1) > 1:
            L, R = time_L, time_R

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
