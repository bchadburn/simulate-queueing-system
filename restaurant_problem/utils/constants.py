PERFORMANCE_DECAY_RATE = .03
SIMULATION_SAMPLES = 50

RESTAURANT_PARAMETERS = {
    'restaurant_0': {
        'hours_opened': 12,
        'menu_time_distribution': {'item_1': {'time': 5, 'proportion': 1 / 6},
                                   'item_2': {'time': 10, 'proportion': 1 / 6},
                                   'item_3': {'time': 15, 'proportion': 1 / 6},
                                   'item_4': {'time': 20, 'proportion': 1 / 6},
                                   'item_5': {'time': 25, 'proportion': 1 / 6},
                                   'item_6': {'time': 30, 'proportion': 1 / 6}},
        'daily_schedule': [{'start_time_minutes': 0, 'num_hourly_orders': 50},
                           {'start_time_minutes': 4 * 60, 'num_hourly_orders': 75},
                           {'start_time_minutes': 8 * 60, 'num_hourly_orders': 100},
                           {'start_time_minutes': 12 * 60, 'num_hourly_orders': None}]
    }
}
