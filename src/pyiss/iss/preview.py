from __future__ import annotations

import base64
import html
import io
import warnings

import pandas as pd
import requests

from IPython.display import HTML, display

from ..opus import preview_url

_SIZE_ALIASES = {
    "thumb": "thumb",
    "thumbnail": "thumb",
    "small": "small",
    "medium": "medium",
    "med": "medium",
    "large": "full",
    "full": "full",
}

_LAYOUT_ALIASES = {
    "grid": "grid",
    "row": "row",
}


def normalize_image_size(image_size: str) -> str:
    size = str(image_size).strip().lower()
    if size not in _SIZE_ALIASES:
        allowed = ", ".join(sorted(_SIZE_ALIASES))
        raise ValueError(f"Unsupported image_size '{image_size}'. Allowed: {allowed}")
    return _SIZE_ALIASES[size]


def normalize_layout(layout: str) -> str:
    key = str(layout).strip().lower()
    if key not in _LAYOUT_ALIASES:
        allowed = ", ".join(sorted(_LAYOUT_ALIASES))
        raise ValueError(f"Unsupported layout '{layout}'. Allowed: {allowed}")
    return _LAYOUT_ALIASES[key]


def normalize_clip_percentiles(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        low_raw, high_raw = value
    except Exception as exc:
        raise ValueError("clip_percentiles must be a (low, high) pair.") from exc

    low = float(low_raw)
    high = float(high_raw)
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
    return low, high


def _stretched_data_uri(url: str, *, clip_percentiles: tuple[float, float]) -> str:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Local intensity stretch requires numpy and Pillow.") from exc

    r = requests.get(url)
    r.raise_for_status()

    arr = np.asarray(Image.open(io.BytesIO(r.content)).convert("F"), dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Image has no finite pixels for stretch.")

    low, high = np.percentile(finite, [clip_percentiles[0], clip_percentiles[1]])
    if not high > low:
        high = low + 1.0

    stretched = np.clip((arr - low) / (high - low), 0.0, 1.0)
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)

    out = Image.fromarray((stretched * 255.0).astype("uint8"), mode="L")
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _resolve_image_src(
    url: str,
    *,
    opusid: str,
    clip_percentiles: tuple[float, float] | None,
) -> str:
    if clip_percentiles is None:
        return url
    try:
        return _stretched_data_uri(url, clip_percentiles=clip_percentiles)
    except Exception as exc:
        warnings.warn(
            f"Could not apply local intensity stretch for {opusid} ({exc}); using raw preview URL.",
            stacklevel=2,
        )
        return url


def gallery_html(
    set_df: pd.DataFrame,
    *,
    image_size: str = "full",
    image_calibrated: bool = False,
    layout: str = "grid",
    intensity: str | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> str:
    size = normalize_image_size(image_size)
    normalized_layout = normalize_layout(layout)
    normalized_clip = normalize_clip_percentiles(clip_percentiles)
    intensity_key = str(intensity).strip().upper() if intensity is not None else "DN"
    if normalized_clip is None and intensity_key != "DN":
        normalized_clip = (1.0, 99.0)

    items = []
    for _, row in set_df.iterrows():
        opusid = str(row["opusid"])
        url = preview_url(opusid, image_size=size, image_calibrated=image_calibrated)
        src = _resolve_image_src(url, opusid=opusid, clip_percentiles=normalized_clip)
        filt = row.get("COISSfilter", "")
        t = row.get("time1", "")
        items.append((opusid, str(filt), str(t), str(src)))

    if not items:
        return "<div style='font-family:monospace;'>No rows to display.</div>"

    if normalized_layout == "row":
        container_style = "display:flex;flex-wrap:nowrap;gap:14px;overflow-x:auto;align-items:flex-start;"
    else:
        container_style = "display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;"

    html_out = ""
    if intensity is not None:
        if normalized_clip is None:
            stretch_txt = "stretch=none"
        else:
            stretch_txt = f"stretch={normalized_clip[0]:g}-{normalized_clip[1]:g}% (local)"
        html_out += (
            "<div style='font-family:monospace;font-size:12px;margin-bottom:8px;'>"
            f"intensity={html.escape(str(intensity))}; "
            f"image_size={html.escape(size)}; "
            f"image_calibrated={str(bool(image_calibrated)).lower()}; "
            f"{html.escape(stretch_txt)}"
            "</div>"
        )
    html_out += f"<div style='{container_style}'>"
    for opusid, filt, t, url in items:
        html_out += f"""
        <figure style="margin:0;width:420px;flex:0 0 auto;">
          <div style="font-family:monospace;font-size:12px;line-height:1.3;margin-bottom:6px;">
            <div><b>{html.escape(opusid)}</b></div>
            <div>{html.escape(filt)} - {html.escape(t)}</div>
          </div>
          <img src="{html.escape(url)}" style="width:100%;height:auto;border-radius:6px;"/>
        </figure>
        """
    html_out += "</div>"
    return html_out


def gallery(
    set_df: pd.DataFrame,
    *,
    image_size: str = "full",
    image_calibrated: bool = False,
    layout: str = "grid",
    intensity: str | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> None:
    display(
        HTML(
            gallery_html(
                set_df,
                image_size=image_size,
                image_calibrated=image_calibrated,
                layout=layout,
                intensity=intensity,
                clip_percentiles=clip_percentiles,
            )
        )
    )
