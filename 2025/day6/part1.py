"""
Author: Darren
Date: 01/12/2025

Solving https://adventofcode.com/2025/day/6

Edited by: Gamer 1

Make sure to install dazbo_commons and use /inputs/input.txt!

"""
import logging
import math
import sys
import textwrap

import dazbo_commons as dc  # For locations
from rich.logging import RichHandler

import aoc_common.aoc_commons as ac  # General AoC utils

# Set these to the current puzzle
YEAR = 2025
DAY = 6

locations = dc.get_locations(__file__)

# Configure root logger with Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(message)s",
    datefmt='%H:%M:%S',
    handlers=[RichHandler(
        rich_tracebacks=True, 
        show_path=False,
        markup=True,
        show_time=False  # Disable Rich's time since we're using our own
    )]
)
logger = logging.getLogger(locations.script_name)
logger.setLevel(logging.DEBUG)

class Puzzle:
    def __init__(self, operator: str):
        self.numbers = []
        self.operator = operator

    def add_number(self, number: int):
        self.numbers.append(number)

    def calculate(self):
        match self.operator:
            case "+":
                return sum(self.numbers)
            case "*":
                return math.prod(self.numbers)
            case _:
                raise ValueError(f"Invalid operator: {self.operator}")
    
    def __str__(self):
        return self.operator + ": " + ",".join(map(str, self.numbers))

def part1(data: list[str]):
    puzzles = []
    operator_indices = {}
    for idx, operator in enumerate(data[-1]):
        if operator in "*/+-":
            operator_indices[idx] = operator
            puzzles.append(Puzzle(operator))

    logger.debug(operator_indices)
    for line in data[:-1]:
        indices_list = list(operator_indices.keys())
        start = end = None
        for puzzle_num, puzzle in enumerate(puzzles):
            if puzzle_num == len(puzzles) - 1: # If last puzzle
                end = len(line)
            else:
                end = indices_list[puzzle_num + 1]
            start = indices_list[puzzle_num]
            logger.debug(line[start:end])
            puzzle.add_number(int(line[start:end]))

    return sum(puzzle.calculate() for puzzle in puzzles)

def main():
    try:
        ac.write_puzzle_input_file(YEAR, DAY, locations)
        with open(locations.input_file, encoding="utf-8") as f:
            input_data = f.read().splitlines() # Most puzzles are multiline strings
            logger.debug(dc.top_and_tail(input_data))
    except (ValueError, FileNotFoundError) as e:
        logger.error("Could not read input file: %s", e)
        return 1

    # Part 1 tests
    logger.setLevel(logging.DEBUG)
    sample_inputs = []
    sample_inputs.append(textwrap.dedent("""\
        123 328  51 64 
         45 64  387 23 
          6 98  215 314
        *   +   *   +  """))
    sample_answers = [4277556]
    test_solution(part1, sample_inputs, sample_answers)

    # Part 1 solution
    logger.setLevel(logging.INFO)
    with ac.timer():
        logger.info(f"Part 1 soln={part1(input_data)}")
    
def test_solution(soln_func, sample_inputs: list, sample_answers: list):
    """
    Tests a solution function against multiple sample inputs and expected answers.

    Args:
        soln_func: The function to be tested (e.g., part1 or part2).
        sample_inputs: A list of sample input strings.
        sample_answers: A list of expected answers corresponding to the sample inputs.

    Raises:
        AssertionError: If any of the test cases fail validation.
    """
    for curr_input, curr_ans in zip(sample_inputs, sample_answers):
        try:
            ac.validate(soln_func(curr_input.splitlines()), curr_ans)
        except AssertionError as e:
            logger.error(f"{soln_func.__name__} test failed: {e}")
            sys.exit(1)
    logger.info(f"{soln_func.__name__} tests passed")
    
if __name__ == "__main__":
    main()
