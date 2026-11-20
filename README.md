# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/qmanank23/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/qmanank23/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase at 2026-11-26 00:00 UTC — a week after launch — as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## How it works

`main.py` (Python 3.13, standard library only) reads one case JSON and, per trophy, fits a weighted least-squares line of logit(earn rate) against ln(1 + hours since the first read) — the adoption curve's shape: earners arrive fast at first and slow down, while the tracked-player base keeps growing and dilutes slow trophies. The slope is shrunk toward its tier's median slope, projected growth is capped at the tier's 90th-percentile growth, and tier backstop caps keep platinum/gold honest in week one. The spread combines a random-walk term with each trophy's residual volatility plus a floor, so the band stays honest through Exophase's refresh shocks. Output is deterministic: same input bytes give the same output bytes.

On a synthetic backtest (200 trophies, logistic growth curves plus common refresh shocks, 4.6-day horizon, the competition's exact pinball loss), this beats the starter persistence baseline by 24% on log-time truth and 60% on logistic-in-linear-time truth.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ejcemhvm&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
