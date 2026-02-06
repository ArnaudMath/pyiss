# pyiss

OPUS-backed helper tools for Cassini ISS.

v0.1 focus: infer “ISS filter sets” around a seed OPUS ID and preview the images.

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

iss_set = infer_set(seed)

iss_set.size
iss_set.available_filters
iss_set.time_window

# Canonical table (no dt_* columns)
df = iss_set.df

# Row helpers
iss_set.row("all")
iss_set.row("CL1")
iss_set.time1("CL1")
iss_set.opusid("CL1")
iss_set.target("CL1")

# Preview images
iss_set.show("all")
```

**Diagnostics**
Use the diagnostic window to inspect gaps across the set boundaries.
```python
win = iss_set.diagnostic_window()

# First set row has dt_prev_s filled from the last neighbour (row A)
# Last set row has dt_next_s filled from the next neighbour (row Z)
win[["_role", "opusid", "time1", "dt_prev_s", "dt_next_s"]]
```

**Low-level DataFrame API**
```python
from pyiss.iss.sets import infer_set_dfs

neigh_df, set_df = infer_set_dfs(seed)
```
