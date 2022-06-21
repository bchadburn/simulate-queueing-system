import pytest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from restaurant_problem.main import merge_times


@pytest.mark.parametrize('menu, result', [({'item_1': {'time': 5, 'proportion': 1 / 6},
                                            'item_2': {'time': 5, 'proportion': 1 / 6},
                                            'item_3': {'time': 15, 'proportion': 1 / 6},
                                            'item_4': {'time': 20, 'proportion': 1 / 6},
                                            'item_5': {'time': 25, 'proportion': 1 / 6},
                                            'item_6': {'time': 30, 'proportion': 1 / 6}},
                                           {5: 0.3333333333333333,
                                            15: 0.16666666666666666,
                                            20: 0.16666666666666666,
                                            25: 0.16666666666666666,
                                            30: 0.16666666666666666}),
                                          ({'item_1': {'time': 5, 'proportion': 1 / 5},
                                            'item_3': {'time': 15, 'proportion': 1 / 5},
                                            'item_4': {'time': 20, 'proportion': 1 / 5},
                                            'item_5': {'time': 25, 'proportion': 1 / 5},
                                            'item_6': {'time': 30, 'proportion': 1 / 5}},
                                           {5: 0.2,
                                            15: 0.2,
                                            20: 0.2,
                                            25: 0.2,
                                            30: 0.2})])
def test_merge_times(menu, result):
    grouped_menu_distributions = merge_times(menu)
    proportion_total = 0
    for proportion in grouped_menu_distributions.values():
        proportion_total += proportion
    assert result == grouped_menu_distributions
