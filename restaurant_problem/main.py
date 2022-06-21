import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import logging
import argparse
from typing import Union
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from restaurant_problem.utils.constants import RESTAURANT_PARAMETERS, PERFORMANCE_DECAY_RATE, SIMULATION_SAMPLES
from restaurant_problem.utils.utils import time_run


def merge_times(menu_distribution):
    """Combines menu items with the same cook time. Sums proportions of individual items of the same cook time."""
    grouped_menu_distribution = {}
    for _, v in menu_distribution.items():
        if v['time'] in grouped_menu_distribution:
            grouped_menu_distribution[v['time']] += v['proportion']
        else:
            grouped_menu_distribution[v['time']] = v['proportion']
    return grouped_menu_distribution


def create_lookup_lists(grouped_distribution: dict):
    """Creates a sorted list of menu times and lookup table for taking samples"""
    menu_time_list = sorted(grouped_distribution.keys())
    lookup_list = []
    running_value = 0
    for time in menu_time_list:
        running_value += grouped_distribution[time]
        lookup_list.append(running_value)
    return menu_time_list, lookup_list


def create_distribution_list(menu_distribution):
    """Creates a sorted list of menu cook times (e.g. [5, 10] 1st item takes 5 minutes to cook)
    and a cumulative proportions list (e.g. [.10, .20] 1st and 2nd item represents 10% of orders). Used to
    to sample menu items based on proportions. e.g. if random number generator return .15, it would return 2nd item"""
    grouped_distribution = merge_times(menu_distribution)
    time_list, lookup_list = create_lookup_lists(grouped_distribution)
    return time_list, lookup_list


def estimate_total_orders(restaurant: dict) -> int:
    """Returns expected total number of orders for one day"""
    daily_schedule = RESTAURANT_PARAMETERS[restaurant]['daily_schedule']
    total_orders = 0
    for idx in range(1, len(daily_schedule)):
        minutes = daily_schedule[idx]['start_time_minutes'] - daily_schedule[idx - 1]['start_time_minutes']
        total_orders += daily_schedule[idx - 1]['num_hourly_orders'] * (minutes / 60)
        idx += 1
    assert daily_schedule[idx - 1]['start_time_minutes'] / 60 == RESTAURANT_PARAMETERS[restaurant][
        'hours_opened'], "Final start time for daily schedule must be equal to hours opened"
    return total_orders


class RestaurantSimulation:
    def __init__(self, schedule, time_list, lookup_list, cooks, decay_rate):
        self.daily_schedule = schedule
        self.lookup_list = lookup_list
        self.max_value = np.max(lookup_list)
        self.time_list = time_list
        self.cooks = cooks
        self.decay_rate = decay_rate

        self.total_orders = 0
        self.completed_orders = 0
        self.clock = 0
        self.order_time = 0
        self.number_in_queue = 0
        self.total_queued = 0
        self.total_queued_time = 0
        self.total_cook_time = 0

        self.cook_queue = None

    def calc_cook_time(self):
        lookup_val = random.uniform(0, self.max_value)
        for idx in range(len(self.lookup_list)):
            if lookup_val < self.lookup_list[idx]:
                return self.time_list[idx]

    def _create_cook_queue(self):
        """Creates initial queue which will be used to track the status of each cook"""
        self.cook_queue = {}
        idx = 0
        for sv in range(self.cooks):
            self.cook_queue[idx] = {
                'occupied': False,
                'time_completed': float('inf'),
                'cook_time_sum': 0  # Total time to complete que
            }
            idx += 1

    def _get_que_status(self):
        """Get current status of each cook"""
        if not self.cook_queue:
            self._create_cook_queue()
        min_time = float('inf')
        status_list = []
        idx_not_busy, min_time_idx = 0, 0
        for queue_idx in self.cook_queue:
            if self.cook_queue[queue_idx]['time_completed'] < min_time:
                min_time_idx = queue_idx
                min_time = self.cook_queue[queue_idx]['time_completed']
            status_list.append(self.cook_queue[queue_idx]['occupied'])
            if not self.cook_queue[queue_idx]['occupied']:
                idx_not_busy = queue_idx  # Select first available que
        busy = all(status_list)
        return min_time, busy, idx_not_busy, min_time_idx

    def timing_routine(self, kitchen_closed=False):
        """Uses 'next-event time advance' (NETA) method for moving clock forward."""
        min_completed, is_busy, idx_not_busy, min_time_idx = self._get_que_status()
        if not kitchen_closed:
            time_next_event = min(self.order_time, min_completed)
        else:
            time_next_event = min_completed
            self.order_time = float('inf')
        self.total_queued_time += self.number_in_queue * (time_next_event - self.clock)
        self.clock = time_next_event

        if self.order_time < min_completed:
            self._new_order(is_busy, idx_not_busy)
        else:
            self.completed_orders += 1
            self._update_cook_queue(min_time_idx)

    def _new_order(self, is_busy, idx_not_busy):
        """Updated queue based on arrival of new order"""
        self.total_orders += 1
        logging.debug(f"number in queue, {self.number_in_queue}, is_busy: {is_busy}, cook_queue: {self.cook_queue}")
        if self.number_in_queue == 0:  # If no other items in queue
            if is_busy:
                self.number_in_queue += 1
                self.total_queued += 1

                # Generate random customer arrival time (poisson distribution)
                for idx in range(0, len(self.daily_schedule)):
                    if self.clock < self.daily_schedule[idx]['start_time_minutes']:
                        self.order_time = self.clock + random.expovariate(self.daily_schedule[0][
                                                                                'num_hourly_orders'] / 60)  # Poisson random sample based on avg hourly order rate.
                        break
                logging.debug(
                    f"queue_idx: {idx_not_busy}, current_cook_queue: {self.cook_queue[idx_not_busy]['occupied']}")

            else:
                # Generate random menu cook time from uniform distribution (5,30)
                cook_time = self.calc_cook_time()
                cook_time = cook_time / (1 - self.cooks * self.decay_rate)
                self.cook_queue[idx_not_busy]['cook_time_sum'] += cook_time
                self.total_cook_time += cook_time
                self.cook_queue[idx_not_busy]['time_completed'] = self.clock + cook_time

                # Generate random customer arrival time for next observation (assumes poisson distribution)
                for idx in range(0, len(self.daily_schedule)):
                    if self.clock < self.daily_schedule[idx]['start_time_minutes']:
                        self.order_time = self.clock + random.expovariate(self.daily_schedule[0][
                                                                                'num_hourly_orders'] / 60)  # Poisson random sample based on avg hourly order rate.
                        break
                self.cook_queue[idx_not_busy]['occupied'] = True
                logging.debug(
                    f"queue_idx: {idx_not_busy}, current_cook_queue: {self.cook_queue[idx_not_busy]['occupied']}")

        else:
            self.number_in_queue += 1
            self.total_queued += 1
            self.order_time = self.clock + random.expovariate(
                self.daily_schedule[0][
                    'num_hourly_orders'] / 60)  # Poisson random sample based on avg hourly order rate.

    def _update_cook_queue(self, min_time_idx):
        """Updates cook queue after a completed order"""
        if self.number_in_queue > 0:
            cook_time = self.calc_cook_time()
            self.cook_queue[min_time_idx]['cook_time_sum'] += cook_time
            self.total_cook_time += cook_time
            self.cook_queue[min_time_idx]['time_completed'] = self.clock + cook_time
            self.number_in_queue -= 1
        else:
            self.cook_queue[min_time_idx]['time_completed'] = float('inf')
            self.cook_queue[min_time_idx]['occupied'] = False


@time_run
def _replicate_simulation(daily_schedule, menu_time_list, lookup_list, cooks, hours_opened,
                          restaurant, num_cooks, simulation_samples, decay_rate) -> pd.DataFrame:
    results = pd.DataFrame(
        columns=['restaurant', 'num_cooks', 'avg_wait_time', 'avg_cook_time', 'utilization',
                 'queued_orders', 'completed_orders', 'kitchen_hrs_opened', 'after_hrs_time'])

    for i in tqdm(range(simulation_samples)):
        np.random.seed(i)
        simulator = RestaurantSimulation(daily_schedule, menu_time_list, lookup_list, cooks, decay_rate)
        while simulator.clock <= hours_opened * 60:
            simulator.timing_routine()
        while simulator.completed_orders < simulator.total_orders:
            simulator.timing_routine(kitchen_closed=True)

        sample_results = pd.Series([restaurant, num_cooks, (
                simulator.total_queued_time / simulator.completed_orders + simulator.total_cook_time / simulator.completed_orders),
                                    simulator.total_cook_time / simulator.completed_orders,
                                    simulator.total_cook_time / simulator.clock, simulator.total_queued,
                                    simulator.completed_orders, hours_opened,
                                    simulator.clock - hours_opened * 60], index=results.columns)
        results = results.append(sample_results, ignore_index=True)
    return results


def find_optimal_num_cooks(simulation_results, max_cooks, daily_schedule, menu_time_list, lookup_list, hours_opened,
                           restaurant, simulation_samples, decay_rate) -> pd.DataFrame:
    num_cooks = int(max_cooks)
    recent_avg_time = hours_opened * 60  # Initiate at max wait time
    min_avg_time = hours_opened * 60  # Initiate at max wait time
    while min_avg_time == recent_avg_time and num_cooks > 0:  # Keep adding cooks until avg cook time doesn't go down
        results_tbl = _replicate_simulation(daily_schedule, menu_time_list, lookup_list, num_cooks, hours_opened,
                                            restaurant, num_cooks, simulation_samples, decay_rate)
        recent_avg_time = np.mean(results_tbl['avg_wait_time'])
        min_avg_time = min(min_avg_time, np.mean(results_tbl['avg_wait_time']))
        logging.info(f'simulations completed for {num_cooks} cooks')
        simulation_results = simulation_results.append(results_tbl, ignore_index=True)
        num_cooks -= 1

    logging.info(f'{restaurant} simulations completed')
    return simulation_results


def extract_restaurant_params(restaurant: dict):
    daily_schedule = RESTAURANT_PARAMETERS[restaurant]['daily_schedule']
    menu_time_distribution = RESTAURANT_PARAMETERS[restaurant]['menu_time_distribution']
    hours_opened = RESTAURANT_PARAMETERS[restaurant]['hours_opened']
    proportion_total = 0
    for proportion in menu_time_distribution.values():
        proportion_total += proportion['proportion']
    assert round(proportion_total, 2) == 1, f"Menu item proportions don't sum to 1, total proportion = {proportion_total}"
    return daily_schedule, menu_time_distribution, hours_opened


def get_max_cooks(restaurant: dict, hours_opened: Union[int, str], avg_menu_cook_time: Union[int, str]) -> int:
    total_orders = estimate_total_orders(restaurant)
    max_orders_per_cook = (hours_opened * 60) / avg_menu_cook_time
    max_cooks = np.ceil(total_orders / max_orders_per_cook)
    return max_cooks


def main(simulation_samples, performance_decay_rate):
    simulation_results = pd.DataFrame(
        columns=['restaurant', 'num_cooks', 'avg_wait_time', 'avg_cook_time', 'utilization',
                 'queued_orders', 'completed_orders', 'kitchen_hrs_opened', 'after_hrs_time'])
    for restaurant in RESTAURANT_PARAMETERS:
        logging.info(f' Starting simulations for {restaurant}')

        # Extract restaurant params
        daily_schedule, menu_time_distribution, hours_opened = extract_restaurant_params(restaurant)

        # Create menu_time and lookup list for generating random menu samples
        menu_time_list, lookup_list = create_distribution_list(menu_time_distribution)
        simulation = RestaurantSimulation(daily_schedule, menu_time_list, lookup_list,
                                          1, performance_decay_rate)  # Initiating class to access calc_cook_time function
        avg_menu_cook_time = np.mean([simulation.calc_cook_time() for x in range(1000)])

        # Select number of cooks to begin simulations
        max_cooks = get_max_cooks(restaurant, hours_opened, avg_menu_cook_time)  # Cooks if there was 100% utilization

        # Run simulations to find optimal number of cooks
        simulation_results = find_optimal_num_cooks(simulation_results, max_cooks, daily_schedule, menu_time_list,
                                                    lookup_list, hours_opened, restaurant, simulation_samples, performance_decay_rate)

    cook_avg_wait_time = simulation_results.groupby(['num_cooks'])['avg_wait_time'].mean()
    logging.info(f'Optimal number of cooks: {cook_avg_wait_time.idxmin()}')
    simulation_results.to_csv('simulation_results.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--simulation_samples', type=int, default=SIMULATION_SAMPLES,
                        help='Number of simulations to run')
    parser.add_argument('-p', '--performance_decay_rate', type=int, default=PERFORMANCE_DECAY_RATE,
                        help='Rate at which performance declines as cooks are added')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if args.debug:
        print('Saving logs to simulation.log')
        logging.basicConfig(filename='simulations.log', filemode='w', level=logging.DEBUG)
        args.simulation_samples = 2
    else:
        logging.basicConfig(level=logging.INFO)

    main(args.simulation_samples, args.performance_decay_rate)
