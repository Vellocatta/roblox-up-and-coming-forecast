#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (hourly/archive snapshots of Roblox's Up-and-Coming
chart, oldest first) and /input/task.json, and writes one forecast to
/output/forecast.json: for each of the top 10 places, a probability for each game
named being at that place, plus the top 10 in order.

Method (standard library only, deterministic, no network, no clock):
  1. For every game ever seen, build its CURRENT place distribution P0 from the
     recent reads (recency-weighted histogram over places 1..10 and rank bands,
     plus ABSENT when it is not on the chart).
  2. Estimate a forward transition table T from the history itself: over every
     (origin read, later read within 72h) pair, how often a game in a given state
     (place 1..10, rank band, or absent) is at each place 1..10 at the later read.
     T is the average place distribution over the 72h window that follows an
     origin, which is exactly what the single forecast is scored on.
  3. Forecast P(game, place) = sum over states of P0(game, state) * T(state, place).
     Two per-game tilts from the freshest playing counts, both fitted on the
     practice window: a playing tilt (games with a large absolute player base
     are coiled springs even at rank 11-50, since the chart is sorted by
     growth, not by playing) and a growth tilt (playing that jumped in the
     last read tends to keep climbing). A rank-velocity momentum tilt was
     tested and dropped: it hurt on the practice window.
     Per place, keep the games with non-negligible probability (up to 80 rows),
     scaled so the place sums to 0.90, leaving an unnamed reserve for games
     never seen before (the rules spread the unnamed share over the chart).
  4. top10 = the 10 games with the highest total top-10 probability, ordered by
     expected place.

The window and places come from task.json, never from the wall clock, so two
runs on the same history write the same forecast byte for byte.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Recency half-life (hours) for the current place histogram.
HALF_LIFE_HOURS = 24.0
# How many recent reads feed the current histogram (also capped by time).
MAX_RECENT_READS = 2
# Forward horizon (hours) the transition table averages over: the scored window.
HORIZON_HOURS = 72.0
# Dirichlet-style smoothing counts added to each transition row, spread by the
# global destination marginal, so no (state, place) is ever exactly zero.
SMOOTH_COUNTS = 2.0
# Per-place probability floor for a named game (below this, leave it unnamed).
P_FLOOR = 0.001
# Most rows named at any one place (rules allow 100).
MAX_ROWS_PER_PLACE = 80
# Target sum of named probabilities per place; the rest stays unnamed.
TARGET_SUM = 0.90
# Temperature applied to candidate probabilities before scaling: >1 sharpens
# (concentrates mass on the most likely games), <1 flattens. 1.0 is honest.
TEMPERATURE = 1.0
# Momentum shift (ranks per unit of rank velocity): a game's place distribution
# is shifted by MOMENTUM_SHIFT * v ranks, where v is its recent rank change
# per read (negative = climbing). 0.0 disables the tilt.
MOMENTUM_SHIFT = 0.0
# Playing tilt: a game's per-place probability is multiplied by
# (1 + PLAYING_GAMMA * playing / max_playing) using the last read's counts.
# The chart is sorted by growth, not by playing, so a game with a huge
# absolute player base sitting at rank 11-50 is a coiled spring.
PLAYING_GAMMA = 2.0
# Growth tilt: multiplied by (1 + GROWTH_GAMMA * clip(log(playing_now /
# playing_prev), -1, 1)). Games whose playing jumped in the last read tend
# to keep climbing over the window.
GROWTH_GAMMA = 3.0


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-07T20:53:19.000Z
    return math.floor(
        __import__("datetime").datetime.fromisoformat(
            s.replace("Z", "+00:00")
        ).timestamp()
    )


def band_of(rank, on_chart):
    if not on_chart:
        return "ABSENT"
    if rank <= 10:
        return rank
    if rank <= 15:
        return "11-15"
    if rank <= 20:
        return "16-20"
    if rank <= 30:
        return "21-30"
    if rank <= 50:
        return "31-50"
    return "51+"


STATES = list(range(1, 11)) + ["11-15", "16-20", "21-30", "31-50", "51+", "ABSENT"]


def load_history(input_dir):
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        reads = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chart = []
            playing = {}
            for g in sorted(row.get("up_and_coming") or [], key=lambda x: x["rank"]):
                chart.append(int(g["universe_id"]))
                if g.get("playing"):
                    playing[int(g["universe_id"])] = g["playing"]
            reads.append({"at": parse_time(row["at"]), "chart": chart, "playing": playing})
    reads.sort(key=lambda r: r["at"])
    return reads


def estimate_transitions(reads):
    """T[state][place 1..10] and T[state]['OUT'], from (origin, forward) pairs."""
    counts = {s: {k: 0.0 for k in list(range(1, 11)) + ["OUT"]} for s in STATES}
    totals = {s: 0.0 for s in STATES}
    marginal = {k: 0.0 for k in list(range(1, 11)) + ["OUT"]}
    all_uids = set()
    for r in reads:
        all_uids.update(r["chart"])
    for i, origin in enumerate(reads):
        origin_rank = {uid: pos + 1 for pos, uid in enumerate(origin["chart"])}
        for j in range(i + 1, len(reads)):
            dt_h = (reads[j]["at"] - origin["at"]) / 3600.0
            if dt_h > HORIZON_HOURS:
                break
            if dt_h <= 0:
                continue
            fwd_top = reads[j]["chart"][:10]
            fwd_set = set(fwd_top)
            for uid in all_uids:
                s = band_of(origin_rank.get(uid, 0), uid in origin_rank)
                if uid in fwd_set:
                    dest = fwd_top.index(uid) + 1
                else:
                    dest = "OUT"
                counts[s][dest] += 1.0
                totals[s] += 1.0
                marginal[dest] += 1.0
    total_pairs = sum(marginal.values()) or 1.0
    marginal = {k: v / total_pairs for k, v in marginal.items()}
    table = {}
    for s in STATES:
        n = totals[s]
        row = {}
        for k in list(range(1, 11)) + ["OUT"]:
            row[k] = (counts[s][k] + SMOOTH_COUNTS * marginal[k]) / (
                n + SMOOTH_COUNTS
            )
        table[s] = row
    return table


def current_distribution(reads):
    """P0[uid][state]: recency-weighted place histogram over recent reads."""
    if not reads:
        return {}
    last_t = reads[-1]["at"]
    recent = reads[-MAX_RECENT_READS:]
    p0 = {}
    for r in recent:
        age_h = max(0.0, (last_t - r["at"]) / 3600.0)
        w = 0.5 ** (age_h / HALF_LIFE_HOURS)
        rank = {uid: pos + 1 for pos, uid in enumerate(r["chart"])}
        seen = set(rank)
        for uid in seen:
            p0.setdefault(uid, {})
            s = band_of(rank[uid], True)
            p0[uid][s] = p0[uid].get(s, 0.0) + w
        # Games on the chart in other recent reads but absent here keep an
        # ABSENT contribution for this read; handled below via the union set.
    # Every game ever seen gets an ABSENT weight for reads where it was absent.
    all_uids = set()
    for r in recent:
        all_uids.update(r["chart"])
    for uid in all_uids:
        dist = p0.get(uid, {})
        for r in recent:
            age_h = max(0.0, (last_t - r["at"]) / 3600.0)
            w = 0.5 ** (age_h / HALF_LIFE_HOURS)
            if uid not in r["chart"]:
                dist["ABSENT"] = dist.get("ABSENT", 0.0) + w
        total = sum(dist.values())
        if total > 0:
            p0[uid] = {s: v / total for s, v in dist.items()}
    return p0


def apply_momentum(p0, reads):
    """Shift each game's place distribution by its recent rank velocity.

    Velocity v is the rank change per read over the last few reads (negative
    means climbing). A game's mass is shifted by MOMENTUM_SHIFT * v ranks:
    climbers move toward better places, fallers toward worse ones. Band
    states are spread uniformly over their ranks before the shift and
    re-binned after; ABSENT mass is left where it is.
    """
    if MOMENTUM_SHIFT == 0.0 or not reads:
        return p0
    window = reads[-4:]
    vel = {}
    for uid in p0:
        pts = []
        for idx, r in enumerate(window):
            if uid in r["chart"]:
                pts.append((idx, r["chart"].index(uid) + 1))
        if len(pts) >= 2:
            (i0, r0), (i1, r1) = pts[0], pts[-1]
            if i1 > i0:
                vel[uid] = (r1 - r0) / (i1 - i0)
    band_lo = {"11-15": 11, "16-20": 16, "21-30": 21, "31-50": 31, "51+": 51}
    band_hi = {"11-15": 15, "16-20": 20, "21-30": 30, "31-50": 50, "51+": 97}
    out = {}
    for uid, dist in p0.items():
        shift = MOMENTUM_SHIFT * vel.get(uid, 0.0)
        if shift == 0.0:
            out[uid] = dist
            continue
        rank_mass = {}
        absent_w = 0.0
        for s, w in dist.items():
            if s == "ABSENT":
                absent_w += w
            elif isinstance(s, int):
                rank_mass[float(s)] = rank_mass.get(float(s), 0.0) + w
            else:
                lo, hi = band_lo[s], band_hi[s]
                per = w / (hi - lo + 1)
                for r in range(lo, hi + 1):
                    rank_mass[float(r)] = rank_mass.get(float(r), 0.0) + per
        new_dist = {}
        if absent_w > 0:
            new_dist["ABSENT"] = absent_w
        for r, w in rank_mass.items():
            nr = r + shift
            if nr < 1.0:
                nr = 1.0
            if nr > 97.0:
                s = "51+"
            else:
                s = band_of(int(math.floor(nr)), True)
            new_dist[s] = new_dist.get(s, 0.0) + w
        total = sum(new_dist.values())
        out[uid] = {s: w / total for s, w in new_dist.items()} if total > 0 else dist
    return out


def playing_tilts(reads):
    """Multiplicative per-game tilt from playing counts in the last reads.

    Returns {uid: multiplier} (1.0 = no tilt), or {} when both tilts are off
    or the counts are missing. Uses the last read's playing relative to the
    chart maximum, plus log growth since the previous read.
    """
    if PLAYING_GAMMA == 0.0 and GROWTH_GAMMA == 0.0:
        return {}
    playing = reads[-1].get("playing") or {}
    if not playing:
        return {}
    mx = max(playing.values())
    prev = reads[-2].get("playing") or {} if len(reads) >= 2 else {}
    out = {}
    for uid, p in playing.items():
        mult = 1.0
        if PLAYING_GAMMA:
            mult *= 1.0 + PLAYING_GAMMA * (p / mx)
        if GROWTH_GAMMA and uid in prev and prev[uid] > 0:
            g = math.log(p / prev[uid])
            g = max(-1.0, min(1.0, g))
            mult *= 1.0 + GROWTH_GAMMA * g
        out[uid] = mult
    return out


def forecast(reads):
    table = estimate_transitions(reads)
    p0 = apply_momentum(current_distribution(reads), reads)
    # P[uid][place] for places 1..10
    prob = {}
    for uid, dist in p0.items():
        row = {k: 0.0 for k in range(1, 11)}
        for s, w in dist.items():
            trow = table[s]
            for k in range(1, 11):
                row[k] += w * trow[k]
        prob[uid] = row
    tilts = playing_tilts(reads)
    if tilts:
        for uid, row in prob.items():
            m = tilts.get(uid, 1.0)
            if m != 1.0:
                for k in row:
                    row[k] *= m
    return prob


def build_output(prob, places):
    probabilities = []
    for k in range(1, places + 1):
        cands = sorted(
            ((p[k], uid) for uid, p in prob.items() if p[k] >= P_FLOOR),
            key=lambda x: (-x[0], x[1]),
        )[:MAX_ROWS_PER_PLACE]
        if TEMPERATURE != 1.0:
            cands = [(p ** TEMPERATURE, uid) for p, uid in cands]
        total = sum(p for p, _ in cands)
        if total <= 0:
            continue
        scale = min(1.0, TARGET_SUM / total)
        for p, uid in cands:
            v = math.floor(p * scale * 1e6) / 1e6
            if v > 0:
                probabilities.append({"universe_id": uid, "place": k, "p": v})
    # top10: highest total top-10 probability, ordered by expected place.
    totals = []
    for uid, p in prob.items():
        tot = sum(p[k] for k in range(1, places + 1))
        if tot <= 0:
            continue
        exp_place = sum(k * p[k] for k in range(1, places + 1)) / tot
        totals.append((tot, exp_place, uid))
    totals.sort(key=lambda x: (-x[0], x[1], x[2]))
    top10 = [uid for _, _, uid in totals[:places]]
    # Order the top10 by expected place (best guess at the order).
    order = {uid: exp for _, exp, uid in totals[:places]}
    top10.sort(key=lambda uid: (order[uid], uid))
    return {"probabilities": probabilities, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    reads = load_history(INPUT_DIR)
    prob = forecast(reads)
    out = build_output(prob, places)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
