from __future__ import annotations

import html
import pandas as pd

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


def gallery_html(
    set_df: pd.DataFrame,
    *,
    image_size: str = "full",
    image_calibrated: bool = False,
    layout: str = "grid",
    intensity: str | None = None,
) -> str:
    size = normalize_image_size(image_size)
    normalized_layout = normalize_layout(layout)
    items = []
    for _, row in set_df.iterrows():
        opusid = str(row["opusid"])
        url = preview_url(opusid, image_size=size, image_calibrated=image_calibrated)
        filt = row.get("COISSfilter", "")
        t = row.get("time1", "")
        items.append((opusid, str(filt), str(t), str(url)))

    if not items:
        return "<div style='font-family:monospace;'>No rows to display.</div>"

    if normalized_layout == "row":
        container_style = "display:flex;flex-wrap:nowrap;gap:14px;overflow-x:auto;align-items:flex-start;"
    else:
        container_style = "display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;"

    html_out = ""
    if intensity is not None:
        html_out += (
            "<div style='font-family:monospace;font-size:12px;margin-bottom:8px;'>"
            f"intensity={html.escape(str(intensity))}; "
            f"image_size={html.escape(size)}; "
            f"image_calibrated={str(bool(image_calibrated)).lower()}"
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
) -> None:
    display(
        HTML(
            gallery_html(
                set_df,
                image_size=image_size,
                image_calibrated=image_calibrated,
                layout=layout,
                intensity=intensity,
            )
        )
    )
