import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.3f}s")

def validate(actual, expected):
    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")

def write_puzzle_input_file(year, day, locations):
    pass
