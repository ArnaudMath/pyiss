from IPython.display import display, HTML
from ..opus import full_preview_url

def gallery(set_df):
    items = []
    for _, row in set_df.iterrows():
        opusid = row["opusid"]
        url = full_preview_url(opusid)
        filt = row.get("COISSfilter", "")
        t = row.get("time1", "")
        items.append((opusid, filt, t, url))

    html = "<div style='display:flex;flex-wrap:wrap;gap:14px;'>"
    for opusid, filt, t, url in items:
        html += f"""
        <figure style="margin:0;width:420px;">
          <div style="font-family:monospace;font-size:12px;line-height:1.3;margin-bottom:6px;">
            <div><b>{opusid}</b></div>
            <div>{filt} — {t}</div>
          </div>
          <img src="{url}" style="width:100%;height:auto;border-radius:6px;"/>
        </figure>
        """
    html += "</div>"
    display(HTML(html))
