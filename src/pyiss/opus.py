import requests
import pandas as pd

from .constants import OPUS_BASE


def to_opus_utc_time(value: object) -> str:
    """
    Convert any parseable datetime-like input to OPUS-friendly time format.
    Value is normalized to UTC, then serialized without timezone suffix.
    Example: 2008-03-01T00:00:00
    """
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception as exc:
        raise ValueError(f"Could not parse OPUS time value '{value}'.") from exc
    if pd.isna(ts):
        raise ValueError("OPUS time value cannot be NaT.")

    base = ts.strftime("%Y-%m-%dT%H:%M:%S")
    if int(ts.microsecond) > 0:
        frac = f"{int(ts.microsecond):06d}".rstrip("0")
        return f"{base}.{frac}"
    return f"{base}"


def _to_int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def data_df(params: dict, cols: list[str]) -> pd.DataFrame:
    """
    Fetch metadata rows from OPUS and auto-page when limit exceeds OPUS page size.

    OPUS paging parameters:
      - startobs: 1-based row index
      - limit: max observations to return for this call
    """
    base_params = dict(params)
    base_params["cols"] = ",".join(cols)

    r = requests.get(f"{OPUS_BASE}/opus/api/data.json", params=base_params)
    r.raise_for_status()
    j = r.json()

    page_rows = list(j.get("page", []))
    out_rows: list[list[object]] = list(page_rows)

    requested_limit = _to_int_or_none(base_params.get("limit"))
    if requested_limit is None or requested_limit <= 0:
        return pd.DataFrame(out_rows, columns=cols)

    first_count = _to_int_or_none(j.get("count")) or len(page_rows)
    available = _to_int_or_none(j.get("available"))
    if available is not None:
        target_total = min(requested_limit, max(available, 0))
    else:
        target_total = requested_limit

    if first_count <= 0 or len(out_rows) >= target_total:
        return pd.DataFrame(out_rows[:target_total], columns=cols)

    start_obs = _to_int_or_none(j.get("start_obs"))
    if start_obs is None:
        start_obs = _to_int_or_none(base_params.get("startobs"))
    if start_obs is None or start_obs < 1:
        start_obs = 1

    # Use observed first-page size as effective per-call cap (often 100 in OPUS).
    per_call_cap = max(1, first_count)
    next_start = start_obs + first_count

    while len(out_rows) < target_total:
        remaining = target_total - len(out_rows)
        page_limit = min(per_call_cap, remaining)
        page_params = dict(base_params)
        page_params["startobs"] = next_start
        page_params["limit"] = page_limit

        r_next = requests.get(f"{OPUS_BASE}/opus/api/data.json", params=page_params)
        r_next.raise_for_status()
        j_next = r_next.json()

        rows_next = list(j_next.get("page", []))
        if not rows_next:
            break

        out_rows.extend(rows_next)

        count_next = _to_int_or_none(j_next.get("count")) or len(rows_next)
        if count_next <= 0:
            break
        next_start += count_next

    return pd.DataFrame(out_rows[:target_total], columns=cols)

def full_preview_url(opusid: str) -> str:
    j = requests.get(f"{OPUS_BASE}/opus/api/image/full/{opusid}.json").json()
    return j["data"][0]["url"]


def preview_url(
    opusid: str,
    *,
    image_size: str = "full",
    image_calibrated: bool = False,
) -> str:
    """
    Resolve the OPUS preview URL for a given image size.

    Notes:
    - `image_calibrated` is passed as a best-effort query parameter because
      OPUS image endpoint capabilities can vary by product.
    - If a calibrated-specific URL key is present, it is preferred.
    """
    size_key = str(image_size).strip().lower()
    candidates = {
        "full": ["full"],
        "medium": ["med", "medium"],
        "med": ["med", "medium"],
        "small": ["small"],
        "thumb": ["thumb", "thumbnail"],
        "thumbnail": ["thumb", "thumbnail"],
    }.get(size_key, [size_key])

    last_http_error = None
    for size in candidates:
        for params in (
            {"image_calibrated": str(bool(image_calibrated)).lower()},
            {},
        ):
            try:
                r = requests.get(f"{OPUS_BASE}/opus/api/image/{size}/{opusid}.json", params=params)
                r.raise_for_status()
            except requests.HTTPError as exc:
                last_http_error = exc
                continue

            j = r.json()
            if not j.get("data"):
                continue

            row = j["data"][0]
            if image_calibrated:
                for key in ("calibrated_url", "url_calibrated", "url"):
                    if key in row and row[key]:
                        return row[key]
            if "url" in row and row["url"]:
                return row["url"]

    if last_http_error is not None:
        raise last_http_error
    raise ValueError(f"OPUS returned no preview URL for {opusid} at size '{image_size}'")
