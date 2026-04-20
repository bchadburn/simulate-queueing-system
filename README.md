# Queueing System Simulation

Monte Carlo simulation to find the optimal number of kitchen staff that minimizes average
customer wait time at a restaurant, accounting for cook performance degradation as team size grows.

## Approach

- Simulates order queues across a range of cook counts
- Models performance degradation per cook as a configurable decay rate
- Runs N simulation samples per configuration to estimate stable averages
- Outputs optimal cook count and wait time distribution

## Results

Default restaurant configuration (12-hour service window, 6-item menu, 50–100 orders/hour peak,
3% performance decay per cook):

| Cook Count | Avg Wait Time | Notes |
|---|---|---|
| < 18 | High | Queue builds faster than it drains during peak |
| **18** | **Minimum** | **Optimal — best wait time vs. degradation tradeoff** |
| > 18 | Increases | Performance decay dominates marginal capacity gain |

Optimal cook count shifts with decay rate and order volume — re-run with `-p` and `-s` flags
to explore the sensitivity.

## Project Structure

```
restaurant_problem/
├── main.py        # Entry point — runs simulation sweep
├── constants.py   # Restaurant parameters (menu, cook times, order rate)
└── ...
tests/             # Pytest suite
```

## Quickstart

Requires Python 3.9+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/bchadburn/simulate-queueing-system.git
cd simulate-queueing-system
uv venv && source .venv/bin/activate
pip install -r requirements.txt
python restaurant_problem/main.py
```

Optional parameters:
```
-s  simulation_samples       Number of simulations per cook count (default: 50)
-p  performance_decay_rate   Cook performance decay as team grows (default: 0.03)
-d  debug                    Debug mode — saves queue logs, limits to 2 samples
```

## Run Tests

```bash
pytest tests/ -v
```
