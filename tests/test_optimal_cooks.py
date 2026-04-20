"""Tests for simulation correctness and optimal cook count sanity bounds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from restaurant_problem.main import (
    create_distribution_list,
    extract_restaurant_params,
    find_optimal_num_cooks,
    get_max_cooks,
)
from restaurant_problem.utils.constants import PERFORMANCE_DECAY_RATE, RESTAURANT_PARAMETERS


def test_optimal_cook_count_within_sensible_range():
    """Optimal cook count for default restaurant params must fall in [1, 30]."""
    restaurant = "restaurant_0"
    daily_schedule, menu_time_distribution, hours_opened = extract_restaurant_params(restaurant)
    menu_time_list, lookup_list = create_distribution_list(menu_time_distribution)

    from restaurant_problem.main import RestaurantSimulation

    sim = RestaurantSimulation(daily_schedule, menu_time_list, lookup_list, 1, PERFORMANCE_DECAY_RATE)
    avg_menu_cook_time = np.mean([sim.calc_cook_time() for _ in range(500)])
    max_cooks = get_max_cooks(restaurant, hours_opened, avg_menu_cook_time)

    simulation_results = pd.DataFrame(
        columns=["restaurant", "num_cooks", "avg_wait_time", "avg_cook_time", "utilization",
                 "queued_orders", "completed_orders", "kitchen_hrs_opened", "after_hrs_time"]
    )
    results = find_optimal_num_cooks(
        simulation_results, max_cooks, daily_schedule, menu_time_list,
        lookup_list, hours_opened, restaurant, simulation_samples=5,
        decay_rate=PERFORMANCE_DECAY_RATE,
    )
    cook_avg_wait = results.groupby("num_cooks")["avg_wait_time"].mean()
    optimal = int(cook_avg_wait.idxmin())
    assert 1 <= optimal <= 30, f"Optimal cook count {optimal} outside expected range [1, 30]"


def test_get_max_cooks_positive():
    """get_max_cooks must return a positive integer for any valid restaurant."""
    restaurant = "restaurant_0"
    params = RESTAURANT_PARAMETERS[restaurant]
    hours_opened = params["hours_opened"]
    max_cooks = get_max_cooks(restaurant, hours_opened, avg_menu_cook_time=17)
    assert max_cooks >= 1
