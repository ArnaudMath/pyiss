from __future__ import annotations

import pandas as pd

from .preview import gallery, gallery_html, normalize_image_size, normalize_layout

_INTENSITY_ALIASES = {
    "DN": "DN",
    "RADIANCE": "RADIANCE",
    "I/F": "I/F",
    "IF": "I/F",
}


def normalize_intensity(value: str) -> str:
    key = str(value).strip().upper()
    if key not in _INTENSITY_ALIASES:
        allowed = ", ".join(sorted(_INTENSITY_ALIASES))
        raise ValueError(f"Unsupported intensity '{value}'. Allowed: {allowed}")
    return _INTENSITY_ALIASES[key]


class ISSSetDisplay:
    """
    Chainable visual preview configuration.
    """

    def __init__(self, rows: pd.DataFrame, *, layout: str = "grid"):
        self._rows = rows.reset_index(drop=True)
        self._image_size = "full"
        self._image_calibrated = False
        self._layout = normalize_layout(layout)
        self._intensity = "DN"

    def _validate_view_config(self) -> None:
        if self._intensity != "DN" and not self._image_calibrated:
            raise ValueError(
                "Intensity modes other than DN require calibrated previews; call image_calibrated(True)."
            )

    def _repr_html_(self) -> str:
        self._validate_view_config()
        return gallery_html(
            self._rows,
            image_size=self._image_size,
            image_calibrated=self._image_calibrated,
            layout=self._layout,
            intensity=self._intensity,
        )

    def image_size(self, value: str) -> "ISSSetDisplay":
        self._image_size = normalize_image_size(value)
        return self

    def image_calibrated(self, value: bool = True) -> "ISSSetDisplay":
        self._image_calibrated = bool(value)
        return self

    def layout(self, value: str) -> "ISSSetDisplay":
        self._layout = normalize_layout(value)
        return self

    def intensity(self, value: str) -> "ISSSetDisplay":
        self._intensity = normalize_intensity(value)
        return self

    def render(self) -> None:
        """
        Explicitly render the gallery (useful outside notebooks).
        """
        self._validate_view_config()
        gallery(
            self._rows,
            image_size=self._image_size,
            image_calibrated=self._image_calibrated,
            layout=self._layout,
            intensity=self._intensity,
        )
