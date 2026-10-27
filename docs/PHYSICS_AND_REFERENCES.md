# PHYSICS_AND_REFERENCES.md — what the students must understand, and what may be cited

> Production bibliography additions and corrections are documented in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and [PROTOCOL_AMENDMENTS.md](PROTOCOL_AMENDMENTS.md). Rodriguez et al. (2025), DOI 10.1029/2024SW004228, and Baker et al. (2019), DOI 10.1029/2019JA027331, are included for the E-only detector decision. The planning review confirmed Sun et al.'s twelve-hour LSTM PE of 0.554 and distinguished Balikhin et al.'s raw-flux from log-flux metrics. Those literature scores are context only; final comparisons use our own measured baselines.

## 1. The causal chain (one screen)
```
coronal hole -> fast solar wind (V up) ──┐
southward IMF Bz ──> reconnection ───────┴─> storm / substorms (Dst down)
        -> seed electrons (tens-hundreds keV) injected into the outer Van Allen belt
        -> chorus waves accelerate them to >2 MeV over ~1–3 days
        -> "killer electrons" at GEO -> deep-dielectric charging -> satellite anomalies
```
Forecasting is possible because of the **1–3 day lag** between cause (solar wind) and effect (MeV flux at GEO); it is hard because the lag varies and chorus-wave strength is not measured upstream. Reference event: St Patrick's Day storm, 15–21 Mar 2015, about 3 d 4 h from trigger to flux peak (from the team's earlier work).

## 2. What each input means
| Feature | Meaning | Why it is there |
|---|---|---|
| `log_flux` | log10 of >2 MeV electron flux at GOES-15 | the current state; flux spans ~4 orders of magnitude so we work in log space |
| `v_sw` | solar-wind speed (km/s) | classic driver (Paulikas–Blake): fast wind → more flux, with a delay |
| `bz_gsm` | interplanetary magnetic field N–S component (nT) | southward (negative) Bz opens the magnetosphere; northward blocks energy entry |
| `dst_nt` | storm-time ring-current index (nT) | negative = geomagnetic storm in progress |
| `sin/cos(LT)` | local time at the satellite | flux has a day–night cycle; a 12 h forecast lands on the opposite phase |

## 3. Verified references (cite only these)
| Ref | Use | Status |
|---|---|---|
| Reeves, G. D. et al. (2011) — DOI 10.1029/2010JA015735 | high Vsw is necessary but not sufficient (flux vs Vsw is a triangle, not a line) | ✅ DOI verified |
| Sun, X. et al. (2024), *J. Space Weather Space Clim.* 14, 25 — DOI 10.1051/swsc/2024021 | LSTM/Transformer for ≥2 MeV GEO flux at 6 h/12 h; Transformer PE 0.828 (6 h), 0.747 (12 h) | ✅ headline numbers; their LSTM 12 h = 0.554 ⚠️ unconfirmed |
| Balikhin, M. A. et al. (2016) — DOI 10.1002/2015SW001303 | REFM vs SNB3GEO daily models; REFM PE −1.31 on raw flux; SNB3GEO PE 0.77 | ✅ |
| Forsyth, C. et al. (2020) — DOI 10.1029/2019SW002416 | threshold-based probabilistic warnings (hit rate 56–83 %) | ✅ |
| Paulikas, G. A. & Blake, J. B. (1979) | original solar-wind-speed vs GEO-flux correlation | ✅ exists; DOI not recorded — look it up before putting in a bibliography |

**Never use:** arXiv 2603.22095 (fabricated by an AI session). **Do not state** any Sun et al. "day-1 onset PE" number (unverified). Verify any link yourself before citing.

## 4. Comparing with prior work — rules
- Our test set (GOES-15, hourly, 2017) differs from Sun et al.'s data and period; **never** write "we match/beat 0.747". Say "for context, Sun et al. report …".
- PE definition used here: `1 − Σ(obs−pred)²/Σ(obs−mean obs)²` on log10 flux. Confirm Sun et al.'s exact definition before putting numbers side by side.
- Our own measured baselines (PLAN §4) are the honest yardstick.
