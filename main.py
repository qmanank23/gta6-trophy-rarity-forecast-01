"""gta6-trophy-rarity-forecast-01 entry: per-trophy saturating-curve forecaster.

For every trophy, work in logit space on the Exophase earn rate (percent,
clamped to [0.01, 99.99]) and fit a weighted least-squares line of
logit(rate) against ln(1 + hours since the first read). The log-time shape
is the adoption curve's shape: a trophy's earners arrive fast at first and
then slow down, while the tracked-player base keeps growing and dilutes
slow trophies (Exophase's launch-night refresh dropped every rate but the
100% one when tracked players jumped 638 -> 3868). A line in log-time
captures both directions: positive slope for trophies players keep earning,
negative slope for trophies dilution outruns.

The slope is shrunk toward its tier's median slope (empirical-Bayes style:
a trophy with few reads inherits its tier's drift) and the projected growth
is capped at its tier's 90th-percentile growth, so one noisy steep slope
cannot launch a trophy past everything its tier-mates did. Absolute caps
per tier are backstops only: platinum means finishing everything, gold
means finishing the story, so in week one they stay low by construction,
while a bronze chapter trophy may run toward saturation.

The spread combines the starter's random-walk term (0.5 logit per
sqrt-day) with each trophy's residual volatility around its own fit and a
floor, so the band stays honest after the refresh shocks the practice
cases show. Quantiles use the standard normal's z-scores, are clipped to
[0.01, 99.99] and rounded to 4 decimals half-up, matching the starter's
rounding convention.

Degenerate input never crashes: a trophy with no usable read falls back to
its tier's median last rate (or 1.0%), slope 0, and a wide spread.

Standard library only, no network, deterministic: same input bytes give
the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of a rate's move, logit units per sqrt-day. Tuned on
# the synthetic backtest (RW 0.30 / floor 0.15 beat the starter's 0.5 on
# both truth families and held ~0.91 coverage on the harder one; tighter
# settings gained <1% on synthetic while risking under-coverage against
# the real refresh shocks, so this stops here).
SIGMA_RW = 0.30
# Floor on the total spread, logit units: refresh shocks move every rate at
# once, so no trophy's band is ever narrower than this.
SPREAD_FLOOR = 0.15
# Cap on the total spread, logit units: wider than this pays at every trophy.
SIGMA_CAP = 2.5
# Shrinkage strength, in reads: a trophy's slope and residual are pulled
# toward its tier's median with this weight.
PRIOR_K = 15.0
# Recency half-life for the slope fit, hours: launch-week movement is fast,
# so reads older than about a day barely count.
HALF_LIFE_HOURS = 12.0
# Absolute backstop caps on the median forecast, percent per tier. Platinum
# is completing everything, gold is finishing the story; in week one both
# stay low, while a bronze chapter trophy may run toward saturation.
ABS_CAP = {"platinum": 15.0, "gold": 40.0, "silver": 75.0, "bronze": 99.5}
# Fallback last rate when a trophy (or its whole tier) has no usable read.
FALLBACK_RATE = 1.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    f = int(math.floor(k))
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def wls_line(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + s*x. Returns (slope, resid_std or None)."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if den < 1e-12:
        return 0.0, None
    slope = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys)) / den
    intercept = ybar - slope * xbar
    n = len(xs)
    if n < 3:
        return slope, None
    dof = max(n - 2, 1)
    var = sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return slope, math.sqrt(var * n / dof)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    days = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])

    # Origin: the first read. x(t) = ln(1 + hours since origin).
    if history:
        origin = parse_time(history[0]["at"])
        last_dt = parse_time(history[-1]["at"])
    else:
        origin = target_dt
        last_dt = target_dt
    x_last = math.log1p(max((last_dt - origin).total_seconds() / 3600.0, 0.0))
    x_target = math.log1p(max((target_dt - origin).total_seconds() / 3600.0, 0.0))
    dx = max(x_target - x_last, 0.0)

    # Per-trophy fit: (slope, resid, n_points, last_rate).
    fits = []
    for i, trophy in enumerate(trophies):
        pts = []  # (x, y, w)
        last_rate = None
        for j, read in enumerate(history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            last_rate = rate
            t = parse_time(read["at"])
            age_h = max((last_dt - t).total_seconds() / 3600.0, 0.0)
            w = 0.5 ** (age_h / HALF_LIFE_HOURS)
            x = math.log1p(max((t - origin).total_seconds() / 3600.0, 0.0))
            pts.append((x, logit(rate), w))
        if len(pts) >= 2:
            slope, resid = wls_line([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        else:
            slope, resid = 0.0, None
        fits.append({"slope": slope, "resid": resid, "n": len(pts), "last": last_rate})

    # Tier pooling: median slope / residual / last rate per tier.
    tiers: dict[str, dict] = {}
    for trophy, fit in zip(trophies, fits):
        t = tiers.setdefault(trophy["tier"], {"slopes": [], "resids": [], "lasts": []})
        if fit["n"] >= 2:
            t["slopes"].append(fit["slope"])
        if fit["resid"] is not None:
            t["resids"].append(fit["resid"])
        if fit["last"] is not None:
            t["lasts"].append(fit["last"])
    for t in tiers.values():
        t["slope_med"] = median(t["slopes"]) if t["slopes"] else 0.0
        t["resid_med"] = median(t["resids"]) if t["resids"] else 0.5
        t["last_med"] = median(t["lasts"]) if t["lasts"] else FALLBACK_RATE

    # Tier growth caps: 90th percentile of the positive per-trophy growth
    # factors over this horizon, so no trophy outruns its tier's own record.
    # Tiers with fewer than 3 fitted trophies borrow the global cap.
    global_growths = []
    for trophy, fit in zip(trophies, fits):
        tier = tiers[trophy["tier"]]
        g = math.exp(fit["slope"] * dx)
        if fit["n"] >= 2 and g > 1.0:
            global_growths.append(g)
    global_cap = percentile(global_growths, 0.9) if global_growths else 1.0
    tier_cap = {}
    for name_tier, t in tiers.items():
        gs = []
        for trophy, fit in zip(trophies, fits):
            if trophy["tier"] != name_tier or fit["n"] < 2:
                continue
            g = math.exp(fit["slope"] * dx)
            if g > 1.0:
                gs.append(g)
        tier_cap[name_tier] = percentile(gs, 0.9) if len(gs) >= 3 else global_cap

    sqrt_days = math.sqrt(max(days, 0.0))
    rows = []
    for trophy, fit in zip(trophies, fits):
        tier = tiers[trophy["tier"]]
        last_rate = fit["last"] if fit["last"] is not None else tier["last_med"]
        # Shrink the slope toward the tier median, by read count.
        n = fit["n"]
        slope_eff = (n * fit["slope"] + PRIOR_K * tier["slope_med"]) / (n + PRIOR_K)
        # Shrink the residual the same way; no fit means the tier median.
        if fit["resid"] is not None:
            resid_eff = (n * fit["resid"] + PRIOR_K * tier["resid_med"]) / (n + PRIOR_K)
        else:
            resid_eff = tier["resid_med"]
        # Project from the last rate, capped at the tier's growth record.
        growth = math.exp(slope_eff * dx)
        cap = tier_cap.get(trophy["tier"], global_cap)
        if growth > 1.0:
            growth = min(growth, max(cap, 1.0))
        lg50 = logit(last_rate) + math.log(growth)
        q50 = inv_logit(lg50)
        # The cap limits growth, never the level: a trophy already above
        # its tier's cap (e.g. 100%) stays where it is.
        q50 = min(q50, max(ABS_CAP.get(trophy["tier"], 99.5), last_rate))
        lg50 = logit(q50)
        # Spread: random walk + residual volatility, both growing with the
        # horizon, plus a floor. Wider when the fit itself was noisy.
        sigma = math.sqrt(
            (SIGMA_RW * sqrt_days) ** 2
            + (resid_eff * sqrt_days) ** 2
            + SPREAD_FLOOR ** 2
        )
        sigma = min(sigma, SIGMA_CAP)
        qs = []
        for z in Z:
            q = inv_logit(lg50 + z * sigma)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Rounding can only break monotonicity by a hair; enforce it.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
