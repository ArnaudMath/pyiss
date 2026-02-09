from __future__ import annotations

import base64
import io

import pandas as pd
import requests
from IPython.display import HTML, display

from ..opus import preview_url
from .display import normalize_intensity
from .preview import normalize_image_size


class ISSRatio:
    """
    Result of an image division operation for one or more paired images.
    """

    def __init__(
        self,
        ratio_arrays: list,
        pair_df: pd.DataFrame,
        *,
        filter_a: str,
        filter_b: str,
        intensity: str,
    ):
        self._ratio_arrays = ratio_arrays
        self._pair_df = pair_df.reset_index(drop=True)
        self._filter_a = filter_a
        self._filter_b = filter_b
        self._intensity = intensity

    @property
    def count(self) -> int:
        return len(self._ratio_arrays)

    @property
    def df(self) -> pd.DataFrame:
        return self._pair_df.copy()

    def summary(self) -> pd.DataFrame:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("summary() requires numpy; install dependency 'numpy'.") from exc

        rows = []
        for i, arr in enumerate(self._ratio_arrays):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                rows.append(
                    {
                        "index": i,
                        "filter_a": self._filter_a,
                        "filter_b": self._filter_b,
                        "ratio_min": float("nan"),
                        "ratio_max": float("nan"),
                        "ratio_mean": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "index": i,
                    "filter_a": self._filter_a,
                    "filter_b": self._filter_b,
                    "ratio_min": float(finite.min()),
                    "ratio_max": float(finite.max()),
                    "ratio_mean": float(finite.mean()),
                }
            )
        return pd.DataFrame(rows)

    def _repr_html_(self) -> str:
        s = self.summary()
        return s.to_html(index=False)

    def show(self, index: int = 0, *, clip_percentiles: tuple[float, float] = (1.0, 99.0)) -> None:
        """
        Render one ratio image as a contrast-stretched grayscale PNG.
        """
        if not (0 <= int(index) < self.count):
            raise IndexError(f"index {index} out of range for {self.count} ratio images")

        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("show() requires numpy and Pillow; install dependencies 'numpy' and 'pillow'.") from exc

        arr = self._ratio_arrays[int(index)]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Ratio image has no finite values.")

        lo, hi = np.percentile(finite, [clip_percentiles[0], clip_percentiles[1]])
        if not hi > lo:
            hi = lo + 1.0
        scaled = (arr - lo) / (hi - lo)
        scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        img = Image.fromarray((scaled * 255.0).astype("uint8"), mode="L")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        meta = self._pair_df.iloc[int(index)]
        html = f"""
        <div style="font-family:monospace;font-size:12px;line-height:1.4;margin-bottom:8px;">
          <div><b>ratio[{int(index)}]</b> {self._filter_a}/{self._filter_b} (intensity={self._intensity})</div>
          <div>{meta['left_opusid']} / {meta['right_opusid']} | dt={meta['dt_s']:.3f}s</div>
        </div>
        <img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;border-radius:6px;"/>
        """
        display(HTML(html))


class ISSPair:
    """
    Strict filter pair object used for scientific arithmetic.
    """

    def __init__(
        self,
        pair_df: pd.DataFrame,
        *,
        filter_a: str,
        filter_b: str,
        image_size: str,
        image_calibrated: bool,
        intensity: str,
        image_type: str,
    ):
        self._pair_df = pair_df.reset_index(drop=True)
        self._filter_a = filter_a
        self._filter_b = filter_b
        self._image_size = normalize_image_size(image_size)
        self._image_calibrated = bool(image_calibrated)
        self._intensity = normalize_intensity(intensity)
        self._image_type = str(image_type).upper()

    @property
    def df(self) -> pd.DataFrame:
        return self._pair_df.copy()

    @staticmethod
    def _require_numpy_pillow():
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Image arithmetic requires numpy and Pillow; add dependencies 'numpy' and 'pillow'."
            ) from exc
        return np, Image

    def _fetch_preview_array(self, opusid: str):
        np, Image = self._require_numpy_pillow()
        url = preview_url(
            str(opusid),
            image_size=self._image_size,
            image_calibrated=self._image_calibrated,
        )
        r = requests.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("F")
        return np.asarray(img, dtype="float64")

    def divide(self, *, epsilon: float = 1e-6) -> ISSRatio:
        """
        Compute left/right image ratios for all strict pairs.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0.")
        if self._image_type != "CAL":
            raise ValueError(
                "divide() requires image_type('CAL') to avoid uncalibrated arithmetic."
            )
        if self._intensity != "DN" and not self._image_calibrated:
            raise ValueError(
                "Non-DN intensity arithmetic requires image_calibrated=True."
            )

        np, _ = self._require_numpy_pillow()
        ratio_arrays = []
        for _, row in self._pair_df.iterrows():
            left = self._fetch_preview_array(str(row["left_opusid"]))
            right = self._fetch_preview_array(str(row["right_opusid"]))
            if left.shape != right.shape:
                raise ValueError(
                    f"Geometry/projection mismatch: {row['left_opusid']} shape {left.shape} != {row['right_opusid']} shape {right.shape}"
                )
            safe_right = np.where(np.abs(right) <= epsilon, np.nan, right)
            ratio_arrays.append(left / safe_right)

        return ISSRatio(
            ratio_arrays,
            self._pair_df,
            filter_a=self._filter_a,
            filter_b=self._filter_b,
            intensity=self._intensity,
        )
