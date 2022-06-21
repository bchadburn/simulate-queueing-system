import functools
import time


def time_run(func):
    """Decorator for printing function runtime"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Completed {func.__name__!r} in {run_time:.2f} secs")
        return result
    return wrapper_timer