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
