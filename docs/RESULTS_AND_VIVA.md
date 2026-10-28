# Results and viva handoff

The LSTM improves on the daily-cycle baseline.

| Forecast | Val PE | Val RMSE | Test PE | Test RMSE |
|---|---:|---:|---:|---:|
| Persistence | 0.358 | 0.912 | 0.290 | 0.864 |
| Persistence + diurnal | 0.707 | 0.617 | 0.680 | 0.580 |
| LSTM | 0.848 ± 0.002 | 0.444 ± 0.003 | 0.815 ± 0.001 | 0.441 ± 0.002 |

1. What do we predict? The change in log10 electron flux twelve hours ahead. Add it to the observed current value to recover the forecast level.
2. What does one sample contain? The last 72 hourly rows, each with log-flux, speed, Bz GSM, Dst, and local-time sine/cosine. Input shape is batch × 72 × 6. Two 64-unit LSTM layers feed a scalar linear head: 51,777 learned parameters.
3. Why E only? Published instrument assessments identify degradation of GOES-15 W. This choice was fixed before training, not selected for a better score.
4. Which time is predicted? A row labelled 10:00 describes 10:00–11:00. Issue at 11:00; predict the mean over 22:00–23:00, ending twelve hours later. Archived processing delay is not modelled.
5. Why not shuffle or normalize on everything? Both risk giving training information from later periods. Statistics use unique training-available hours; both issue and target times must stay in their assigned split.
6. Why preserve gaps? A missing day is not an invitation to shift the next day's timestamps. Only bounded gaps up to three hours are interpolated, and their right endpoints must already be available. Current anchors and labels remain observed.
7. What is the real baseline? Persistence plus the training-only mean twelve-hour change for the current local-time bin. It scores test PE 0.680. Plain persistence is weaker because the target falls on the opposite side of the daily cycle.
8. How did the LSTM do? Test PE 0.815 ± 0.001; the mean difference from the diurnal baseline is +0.135. Explain this measured result without claiming statistical significance.
9. Why three seeds? They expose optimization variability. Their sample SD does not estimate uncertainty over future storms. The plotted model is predetermined seed zero, not the best test seed or an ensemble.
10. How do we know the implementation works? The overfit MSE was 0.007055, checkpoint reload was exact, leakage/timing tests passed, and saved predictions reproduce the scores. Validation selects checkpoints; test is read for scoring only after freeze.
11. What are the limits? One satellite/year, overlapping samples, contaminated operational flux, floor convention, missing-input exclusions, and unmodelled input latency. No operational alert claim and no direct ranking against a different paper's dataset.
12. Why no Transformer? It was optional. The required experiment and complete reproducibility/report were prioritized before opening test results.

Rehearsal: each student should explain one raw observation's path through hourly aggregation, split assignment, interpolation eligibility, normalization, the LSTM tensor, delta reconstruction and scoring. Do not assign personal contributions that cannot be substantiated.
