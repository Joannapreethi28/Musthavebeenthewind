"""Reject malformed source files before they enter the forecasting pipeline."""

import calendar
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

MAX_DATA_BYTES = 8 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_size(path: Path) -> None:
    require(0 < path.stat().st_size <= MAX_DATA_BYTES, "Empty or oversized data file")


def utc_string(milliseconds: float) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def attribute_text(value: object) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def validate_goes(path: Path, year: int, month: int) -> dict:
    """Check the full monthly time grid and both sensors without filling any gaps."""
    check_size(path)
    with path.open("rb") as stream:
        require(stream.read(4) == b"CDF\x01", "Expected NetCDF-3 classic (CDF version 1)")
    expected_rows = calendar.monthrange(year, month)[1] * 24 * 12
    start_ms = datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000
    with netcdf_file(path, "r", mmap=False) as dataset:
        required = ["time_tag"] + [
            sensor + suffix
            for sensor in ("E2E", "E2W")
            for suffix in ("_COR_FLUX", "_QUAL_FLAG")
        ]
        require(all(name in dataset.variables for name in required), "Missing required GOES variables")
        rows = dataset.variables['time_tag'].shape[0]
        require(0 < rows <= expected_rows, 'Unexpected monthly record count')
        arrays = {}
        for name in required:
            variable = dataset.variables[name]
            require(variable.shape == (rows,), f"Unexpected shape for {name}: {variable.shape}")
            arrays[name] = variable[:].astype("float64")
            require(np.isfinite(arrays[name]).all(), f"Nonfinite values in {name}; expected numeric fills")
        units = attribute_text(getattr(dataset.variables["time_tag"], "units", ""))
        require(units.lower().startswith("milliseconds since 1970-01-01 00:00:00"),
                f"Unexpected time units: {units}")
        timestamps = arrays["time_tag"]
        offset = float((timestamps[0] - start_ms) % 300_000)
        # Record a possible interval-centre label instead of silently shifting it.
        positions = (timestamps - start_ms - offset) / 300_000
        require((positions >= 0).all() and (positions < expected_rows).all(), 'GOES dates outside requested month')
        require(np.equal(positions, np.floor(positions)).all() and (np.diff(positions) > 0).all(),
                'GOES dates duplicated, unordered, or off cadence')
        sensors = {}
        for sensor in ("E2E", "E2W"):
            flux, quality = arrays[sensor + "_COR_FLUX"], arrays[sensor + "_QUAL_FLAG"]
            require(((quality >= 0) & (quality == np.floor(quality))).all(), f"Invalid {sensor} quality flags")
            valid = (flux > 0) & (quality == 0)
            require(valid.any(), f"No usable {sensor} observations in month")
            sensors[sensor] = {
                "valid_samples": int(valid.sum()),
                "invalid_samples": int((~valid).sum()),
                "nonzero_quality_samples": int((quality != 0).sum()),
                "fill_flux_samples": int((flux == -99999).sum()),
                "flux_units": attribute_text(getattr(dataset.variables[sensor + "_COR_FLUX"], "units", "")),
            }
    return {
        "format": "NetCDF-3 classic", "rows": rows, "expected_rows": expected_rows,
        "absent_grid_records": expected_rows - rows,
        "first_time_utc": utc_string(timestamps[0]),
        "last_time_utc": utc_string(timestamps[-1]),
        "cadence_seconds": 300, "label_offset_seconds": offset / 1000,
        "time_units": units, "sensors": sensors,
    }


def validate_omni(path: Path, year: int) -> dict:
    """Check all 55 numeric columns and the complete annual UTC calendar."""
    check_size(path)
    # ASCII numeric parsing does not execute any downloaded content.
    values = np.loadtxt(path, dtype="float64", comments=None, encoding="ascii", ndmin=2)
    rows = (366 if calendar.isleap(year) else 365) * 24
    require(values.shape == (rows, 55), f"Expected OMNI shape {(rows, 55)}, got {values.shape}")
    require(np.isfinite(values).all(), "OMNI contains nonfinite values instead of documented numeric fills")
    hours = np.arange(rows)
    require((values[:, 0] == year).all(), "OMNI year does not match filename")
    require(np.array_equal(values[:, 1], hours // 24 + 1), "OMNI days are missing, duplicated, or unordered")
    require(np.array_equal(values[:, 2], hours % 24), "OMNI hours are missing, duplicated, or unordered")
    fields = {}
    for name, column, fill in (("bz_gsm", 16, 999.9), ("v_sw", 24, 9999.), ("dst_nt", 40, 99999.)):
        observed = values[:, column] != fill
        require(observed.any(), f"No observed OMNI {name} values")
        if name == "v_sw":
            require((values[observed, column] > 0).all(), "Nonpositive observed solar-wind speed")
        fields[name] = {"column_1_based": column + 1, "fill_value": fill,
                        "observed_rows": int(observed.sum()), "missing_rows": int((~observed).sum())}
    result = {
        "format": "ASCII numeric, 55 columns", "rows": rows,
        "first_time_utc": f"{year}-01-01T00:00:00+00:00",
        "last_time_utc": f"{year}-12-31T23:00:00+00:00",
        "cadence_seconds": 3600, "fields": fields,
    }
    if year == 2015:
        dst = values[:, 40]
        minimum = float(dst[dst != 99999].min())
        require(minimum == -234 and dst[(76 - 1) * 24 + 22] == -234,
                "2015 Dst sanity check differs from the documented -234 nT at day 76, hour 22; investigate source revision")
        result["dst_sanity_check"] = {"status": "pass", "minimum_nt": minimum, "day_of_year": 76, "hour": 22}
    return result
