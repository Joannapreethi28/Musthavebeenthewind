# VIVA_NOTES.md — for Jabin & Joanna (agents: skim only)

## The whole project in one picture
```
 last 72 hours                                     12 h later
 ┌────────────────────────────┐                   ┌─────────┐
 │ log-flux  V  Bz  Dst  LT   │──► LSTM / Attention ──► Δ̂ ──► log-flux(t) + Δ̂ ──► forecast
 │  (72 rows × 6 columns)     │                   └─────────┘
 └────────────────────────────┘      judged against:  "same as now"  and  "same as now + time-of-day shift"
```

## Questions you must answer cold (short answers)
1. **Why a chronological split?** Forecasting never sees the future. Shuffling lets the model train on days adjacent to test days (neighbouring hours are almost identical) → inflated score that would collapse on real use.
2. **Why normalise with train statistics only?** Using val/test statistics leaks information about the future into training.
3. **Why predict Δ, not the level?** Δ = 0 equals persistence, so the net starts from a strong baseline and only learns the correction; also the flux level drifts across years (mean log10 2.5 → 3.0), Δ is more stable.
4. **Why log10?** Flux spans ~4 orders of magnitude and is roughly log-normal; MSE on raw flux would be dominated by rare spikes.
5. **Why does plain persistence do badly at 12 h?** Flux has a strong day–night cycle (≈0.74 log10 peak-to-trough): 12 h later we are on the opposite phase. Persistence + diurnal (a lookup table!) already reaches PE 0.68. That is the real bar.
6. **What does PE mean?** 1 = perfect, 0 = no better than always predicting the mean of the observations, < 0 = worse than that.
7. **Why are 35k windows not 35k independent samples?** Consecutive windows overlap by 71 of 72 hours; the number of independent storm events is a few hundred. Hence small models, dropout, early stopping, 3 seeds.
8. **Why might a Transformer not beat an LSTM here?** Attention has no built-in sense of order or locality and needs more data; the LSTM's recurrence is a good prior for smooth physical time series. (Sun et al. found the Transformer better at 12 h — so it is an open question, we test it.)
9. **What is overfitting and how do you see it?** Train loss keeps falling while val loss turns upward; early stopping restores the best epoch.
10. **What can't this model do?** One test year, one satellite, one big storm; no uncertainty estimate; inputs end at Earth (Bz, V from L1 give ~1 h lead only; the rest of the skill comes from flux history + time-of-day).

## LSTM in 60 seconds (analogy: a notebook with a pen, an eraser and a reader)
- State per step: hidden `h` (what I report) and cell `c` (the notebook).
- **Forget gate** = eraser: how much of `c` to keep. **Input gate** = pen: how much new info to write. **Output gate** = reader: how much of `c` to show as `h`.
- Because `c` is updated by *adding*, gradients survive many steps (plain RNNs lose them → vanishing gradient).
- Shapes here: input `(B,72,6)` → `h` `(B,72,64)` → take the last step `(B,64)` → Linear → `(B,1)`.

## Self-attention in 60 seconds (analogy: every hour asks a question of every other hour)
- Each hour makes a **query** ("what do I need?"), a **key** ("what do I offer?") and a **value** ("my content").
- `Attention = softmax(Q·Kᵀ / √d) · V` — similarity scores become weights; each hour becomes a weighted blend of all hours.
- Attention itself ignores order, so we **add a positional embedding** (one learned vector per hour 0…71).
- Mean-pool over the 72 outputs → Linear → forecast. Optional extra: plot the attention weights to see whether the model looks ~24 h back.

## Metrics & plots you should be able to produce on demand
Loss curves (train vs val) · forecast vs observed for Sept 2017 · error by local-time hour · results table (baseline vs LSTM vs Transformer, mean ± std over 3 seeds).
