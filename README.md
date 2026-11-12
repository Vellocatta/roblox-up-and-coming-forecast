# Roblox Up-and-Coming Top-10 Forecaster

Entry for the make.fun competition
`roblox-up-and-coming-forecast-01`: forecast Roblox's Up-and-Coming chart
top 10 — a probability for each game at each place — for the 72 hours after
entries close.

## Run

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR/history.ndjson` + `$MAKEFUN_INPUT_DIR/task.json`
(defaults `/input`, `/output`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json`.
Python 3.12+ standard library only, no network, no wall clock: two runs on
the same history produce the same file byte for byte (0.1 s on the kit
history).

## Method

1. **Current distribution P0.** For every game ever seen, a recency-weighted
   histogram over the last 2 reads (24 h half-life) of its state: place 1–10,
   a rank band (11–15, 16–20, 21–30, 31–50, 51+), or ABSENT.
2. **Forward transition table T**, estimated from the history itself: over
   every (origin read, later read within 72 h) pair, how often a game in a
   given state ends up at each place 1–10. T is the average place distribution
   over the 72 h window that follows an origin — exactly what the single
   forecast is scored on. Rows are Dirichlet-smoothed (2.0 counts, spread by
   the global destination marginal).
3. **Forecast** P(game, place) = Σ P0(game, state) · T(state, place), with two
   per-game tilts from the freshest playing counts:
   - **Playing tilt** (×(1 + 2.0 · playing / max playing)): the chart is
     sorted by growth, not by playing, so a game with a huge absolute player
     base sitting at rank 11–50 is a coiled spring.
   - **Growth tilt** (×(1 + 3.0 · clip(log(playing now / playing prev), −1, 1))):
     playing that jumped in the last read tends to keep climbing.
   A rank-velocity momentum tilt was tested and dropped — it hurt on the
   practice window. Per place, the top 80 games with p ≥ 0.001 are named and
   scaled to sum 0.90, leaving a 0.10 unnamed reserve for games never seen
   before. `top10` is the 10 games with the highest total top-10 probability,
   ordered by expected place.
4. **Honest calibration.** Temperature 1.0: sharpening the probabilities
   (temperature > 1) badly hurt the log score on the practice window, so the
   forecast reports its true estimated probabilities.

## Local practice score

Scored against the kit's practice window (10 reads, 2026-11-07 → 2026-11-10)
with the kit's own scoring rule:

| forecaster | log score |
|---|---|
| house persistence baseline | −43.51 |
| this entry | **−34.46** |

The oracle (true window frequencies as the forecast) scores −13.18, so there
is still headroom — but the entry beats the house baseline by ~9 nats on the
practice window.

## Files

- `main.py` — the whole forecaster (stdlib only).
- `.makefun/owner.json` — the make.fun wallet that owns this entry.
