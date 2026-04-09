# Cassini ISS Query Metadata Parameters

This list is tailored for the **Cassini Imaging Science Subsystem (ISS)** in OPUS and focuses on practical query parameters for `pyiss` users.

## Query Style in `pyiss` v0.4.1

For metadata-driven search, prefer the generic parameter pattern:

```python
query.param("key", value)
```

`pyiss` can keep just one convenience method for time windows:

```python
query.time("YYYY-MM-DDTHH:MM:SS", "YYYY-MM-DDTHH:MM:SS")
```

All other OPUS constraints can be expressed through `.param(...)`.

## Type 1: Identification and String Fields

These parameters accept strings (exact or partial matches depending on OPUS behavior).

### `opusid` (OPUS ID)

Unique observation identifier, typically in a form like `co-iss-w1294561143`.

### `bundleid` (Bundle/Volume ID)

PDS archive bundle identifier. For Cassini ISS, commonly ranges from `COISS_2001` to `COISS_2111`.

### `CASSINIobsname` (Observation Name)

String identifying the planned observation sequence.

### `primaryfilespec` (Primary File Specification)

Exact file path in the PDS archive.

## Type 2: Geometry and Range Fields

These expect numeric values. For ranged queries, use lower/upper suffixes (`1` for minimum, `2` for maximum).

Example:

```python
query.param("phase1", 20).param("phase2", 80)
```

### `phase1` / `phase2` (Observed Phase Angle)

- Type: number
- Typical range: `0` to `180` degrees

### `incidence1` / `incidence2` (Observed Incidence Angle)

- Type: number
- Typical range: `0` to `180` degrees

### `emission1` / `emission2` (Observed Emission Angle)

- Type: number
- Typical range: `0` to `180` degrees

### `duration` (Exposure Duration)

- Type: number
- Range: positive values (typically seconds)

### `RINGGEOringradius1` / `RINGGEOringradius2` (Observed Ring Radius)

- Type: number
- Units: commonly `km`

### `resolution1` / `resolution2` (Observed Resolution)

- Type: number
- Typical units: `km/pixel`

## Type 3: Multiple-Choice Fields

These accept constrained values (case-insensitive in OPUS). Multiple values can generally be comma-separated for OR behavior.

### `target` (Intended Target Name)

- `SATURN`
- `TITAN`
- `ENCELADUS`
- `MIMAS`
- `TETHYS`
- `DIONE`
- `RHEA`
- `IAPETUS`
- `HYPERION`
- `PHOEBE`
- `PAN`
- `DAPHNIS`
- `ATLAS`
- `PROMETHEUS`
- `PANDORA`
- `EPIMETHEUS`
- `JANUS`
- `HELENE`
- `TELESTO`
- `CALYPSO`
- `POLYDEUCES`
- `METHONE`
- `ANTHE`
- `PALLENE`
- `AEGAEON`
- `SKY`
- `STAR`
- `CALIBRATION`

### `COISScamera` (ISS Camera)

- `ISSNA` (Narrow Angle Camera)
- `ISSWA` (Wide Angle Camera)

### `COISSfilter` (Spectral Filter)

- `CLEAR`
- `UV1`, `UV2`, `UV3`
- `BL1`, `BL2`
- `GRN`
- `RED`
- `IR1`, `IR2`, `IR3`, `IR4`, `IR5`
- `MT1`, `MT2`, `MT3` (methane bands)
- `CB1`, `CB2`, `CB3` (continuum bands)
- `HAL` (H-alpha)
- `P0`, `P60`, `P120` (polarizers)

### `observationtype` (Observation Type)

- `IMAGE`
- `CALIBRATION`

### `COISSshuttermode` (Shutter Mode)

- `NACONLY`
- `WACONLY`
- `BOTH`
