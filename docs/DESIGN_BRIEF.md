# DESIGN_BRIEF.md — results page & screenshots

Load the `impeccable` skill if your session has it (else `frontend-design`) **before** writing any UI. This brief is the floor.

## 1. Purpose & audience
A static, offline results page that two students screenshot for LinkedIn / GitHub / the course submission. Readers: a recruiter or professor glancing for 20 seconds, then a curious engineer reading closely. It must look like a careful piece of scientific work, **not** a SaaS landing page and **not** an AI-generated template.

## 2. Art direction (decisions already made — follow them)
- **Concept:** an observatory logbook / journal figure plate. The charts are the content; text annotates them.
- **Palette:** warm paper background (~`#F3EFE6`), near-black ink (~`#1B1A17`), one accent for the model forecast (signal vermilion ~`#D9441E`), muted grey-blue for observed data (~`#4C5C6B`), hairline rules ~`#CFC8B8`. Provide a dark variant only if time allows. No gradients, no glow, no glassmorphism, no purple/blue-violet.
- **Type:** a text serif for headings and figure captions (Newsreader or Source Serif 4), a clean grotesque for UI labels (e.g. IBM Plex Sans or Instrument Sans — **not Inter**), a monospace with tabular figures for every number (IBM Plex Mono / JetBrains Mono). Self-host or use system fallbacks so it works offline.
- **Layout:** asymmetric 12-column grid, generous margins, figures wide, captions in a narrow side column. One idea per screen. Max ~1100 px content width.
- **Charts:** direct labels on lines instead of legends; hairline grid; axis in real units (flux as 10ᵏ e/cm²/s/sr, time in UTC dates); annotate the one thing that matters (the storm onset); forecast = accent, observed = ink/grey-blue, baselines = thin neutral lines. Use real data from `results/`, never mocked.
- **Copy:** plain, specific, short. State what the numbers are (test year 2017, 12 h ahead, PE vs baselines). Add a visible "Limitations" footnote. No hype words, no "AI-powered", no emoji, no stock icons.

## 3. Screens (each is a screenshot in `report/screens/`)
| File | Content |
|---|---|
| `01_storm.png` (hero) | Sept 2017 storm window (1–20 Sep, 12-h-ahead forecasts): observed vs persistence+diurnal vs LSTM (vs Transformer if present). Headline numbers beside it: test PE of each model. |
| `02_scoreboard.png` | Table/dot plot: persistence · persistence+diurnal · LSTM [· Transformer] with PE and RMSE (val and test), mean ± std over seeds. A thin marker line at the bar (0.683). |
| `03_diurnal.png` | Two small multiples: mean log-flux vs local time (the day–night cycle), and model error vs local time. |
| `04_method.png` | Method strip: data sources → 72 h window → model → 12 h forecast; a timeline bar of train 2012–15 / val 2016 / test 2017; loss curves (train vs val). |
| `social.png` | 1200×630 share card: the hero chart + one line of headline + names. |
Capture at 1440×900, device scale factor 2, via Playwright (`report/make_screens.py` screenshots each `<section id="s1..s4">` element and the card). Fail loudly if a number on the page does not match `metrics.json`.

## 4. Data contract
`evaluate.py` writes `report/data.json`: `{meta, metrics, storm: {time[], obs[], persistence[], persistence_diurnal[], lstm[], transformer[]?}, diurnal: {lt[], mean_logflux[], err_by_lt:{model:[…]}}, history:{model:{train[], val[]}}}`. The page reads only this file (inline it into the HTML so it works from `file://`).

## 5. Technical constraints
Single `index.html` + `style.css` (+ optional small JS). Hand-written SVG or a small library (uPlot / D3 modules); no build step, no frameworks, no CDN at runtime. Responsive down to 390 px, but the screenshots are the desktop layout. Accessible contrast (≥ 4.5:1 for text), colour is never the only encoding.

## 6. Anti-slop checklist (reject your own work if any is true)
Centered hero with gradient text · three identical rounded cards in a row · purple/blue gradients · emoji or generic icon set · default Inter/system-blue look · drop shadows everywhere · legends instead of direct labels · placeholder or invented numbers · every section the same height and rhythm · stock "Features" section.
