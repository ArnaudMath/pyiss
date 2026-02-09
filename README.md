# pyiss

OPUS-backed helper tools for Cassini ISS.

v0.3 focus: composable ISS set operations plus explicit scientific arithmetic.

## Install

### For users (recommended: install from GitHub)
In a notebook cell:

```python
%pip install "pyiss @ git+https://github.com/ArnaudMath/pyiss.git"
```

**Quickstart**
```python
from pyiss import infer_set

# Seed with any valid ISS OPUS ID
seed = "coiss_2002-03-30_1234"

set0 = infer_set(seed)

set0.size
set0.available_filters
set0.filter_counts
set0.time_window

# Canonical table (no dt_* columns)
df = set0.df

# Row inspection (v0.2 uses generic metadata access)
set0.filter("CL1").metadata("opusid", "time1", "target")

# Strict single-observation selection (always returns exactly one row or raises)
obs0 = set0.select("GRN", which=0)
obs1 = set0.select("GRN", nearest="seed")
obs2 = set0.select("GRN", nearest="time", t="2015-10-14T10:10:00Z")
obs3 = set0.select(opusid=obs0.df.iloc[0]["opusid"])
```

**Composable v0.2 operations**
```python
# Filter-slicing: returns a new ISSSet-like object
set_grn = set0.filter("GRN")
set_mix = set0.filter("GRN", "UV3")

# Visual-only display with chained OPUS-like preview options
# (Notebook renders the final object once; use .render() in scripts)
set0.show("GRN", "UV3", layout="row").image_size("medium").image_calibrated(True)

# Explicit non-visual downstream image type selection
set_cal = set_mix.image_type("CAL")
set_cal.selected_image_type  # "CAL"

# Generic OPUS metadata access (no hardcoded field list)
print(set_mix.metadata("opusid", "time1"))
```

**Scientific v0.3 operations**
```python
# Intensity declaration is explicit (non-DN requires calibrated previews)
set0.show("GRN").image_calibrated(True).intensity("I/F")

# Pair using a pre-selected single observation + a filter
set_cal = set0.image_type("CAL")
obs_ir1 = set_cal.select("IR1", which=0)
pair = set_cal.pair(obs_ir1, "MT3+IRP90", image_size="medium", image_calibrated=True, intensity="DN")
# Optional strict guard:
# pair = set_cal.pair(obs_ir1, "MT3+IRP90", max_dt_s=120, ...)
ratio = pair.divide()

# Numeric summary and visual inspection
ratio.summary()
ratio.show()
```

**Diagnostics**
Use the diagnostic window to inspect gaps across the set boundaries.
```python
win = set0.diagnostic_window()

# First set row has dt_prev_s filled from the last neighbour (row A)
# Last set row has dt_next_s filled from the next neighbour (row Z)
win[["_role", "opusid", "time1", "dt_prev_s", "dt_next_s"]]
```

**Low-level DataFrame API**
```python
from pyiss.iss.sets import infer_set_dfs

neigh_df, set_df = infer_set_dfs(seed)
```
