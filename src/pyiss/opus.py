import requests
import pandas as pd

from .constants import OPUS_BASE

def data_df(params: dict, cols: list[str]) -> pd.DataFrame:
    params = dict(params)
    params["cols"] = ",".join(cols)
    r = requests.get(f"{OPUS_BASE}/opus/api/data.json", params=params)
    r.raise_for_status()
    j = r.json()
    return pd.DataFrame(j["page"], columns=cols)

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
