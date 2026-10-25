"""Throwaway reference — NOT the project pipeline; do not import it.

Reproduces the baseline numbers quoted in PLAN.md section 4 so the real pipeline can be checked against it.
Usage:  python reference/probe_baselines.py <folder containing g15_epead_e13ew_5m_*.nc files for 2012-2017>
(2011 files may be present; they are ignored for training because the satellite was elsewhere.)
Expected output (approx): val2016 persist PE 0.354 / +diurnal 0.706 ; test2017 persist PE 0.293 / +diurnal 0.683
"""
import sys, glob, warnings
import numpy as np
from scipy.io import netcdf_file

warnings.filterwarnings("ignore")
LON_WEST = 135.5  # fixed: the file's west_longitude field is fill (-99999) for all of 2016

T, A, B = [], [], []
for p in sorted(glob.glob(f"{sys.argv[1]}/g15_epead_e13ew_5m_*.nc")):
    f = netcdf_file(p, "r", mmap=False)
    T.append(f.variables["time_tag"][:].astype("float64") / 1000.0)
    for store, k in ((A, "E2W"), (B, "E2E")):
        v = f.variables[k + "_COR_FLUX"][:].astype("float64")
        q = f.variables[k + "_QUAL_FLAG"][:]
        store.append(np.where((v > 0) & (q == 0), v, np.nan))
    f.close()
T, A, B = np.concatenate(T), np.concatenate(A), np.concatenate(B)

comb = np.nanmean(np.vstack([A, B]), axis=0)             # linear mean of the two detectors
hour = np.floor(T / 3600).astype(np.int64)
h0 = hour.min(); idx = hour - h0; n = idx.max() + 1
ok = ~np.isnan(comb)
cnt = np.bincount(idx, weights=ok, minlength=n)
sm = np.bincount(idx, weights=np.nan_to_num(comb), minlength=n)
flux = np.where(cnt >= 6, sm / np.maximum(cnt, 1), np.nan)
lf = np.log10(flux)                                       # NOTE: the real pipeline also floors flux at 1.0

habs = h0 + np.arange(n)                                  # hours since epoch
year = (habs * 3600).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
lt_bin = np.floor(((habs % 24) + 0.5 - LON_WEST / 15.0) % 24).astype(int) % 24

x, y, ltx, yrx = lf[:-12], lf[12:], lt_bin[:-12], year[:-12]
pe = lambda o, p: 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
delta = np.array([np.nanmean((y - x)[(yrx >= 2012) & (yrx <= 2015) & (ltx == k) & np.isfinite(y - x)]) for k in range(24)])
print("mean 12h delta by LT hour (train):", np.round(delta, 2).tolist())
for name, yr in (("val2016", 2016), ("test2017", 2017)):
    m = (yrx == yr) & np.isfinite(x) & np.isfinite(y)
    rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
    print(f"{name}: N={m.sum()}  persist PE={pe(y[m], x[m]):.3f} RMSE={rm((y - x)[m]):.3f}"
          f"  | persist+diurnal PE={pe(y[m], x[m] + delta[ltx[m]]):.3f} RMSE={rm((y - x - delta[ltx])[m]):.3f}")
