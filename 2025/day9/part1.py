"""
Author: Darren
Date: 01/12/2025

Solving https://adventofcode.com/2025/day/7

Edited by: Gamer 1

Make sure to install dazbo_commons and use /inputs/input.txt!

"""

import logging
import sys
import textwrap
from itertools import combinations
from typing import NamedTuple

import dazbo_commons as dc  # For locations
from rich.logging import RichHandler

import aoc_common.aoc_commons as ac  # General AoC utils

# Set these to the current puzzle
YEAR = 2025
DAY = 9

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

class Point(NamedTuple):
    x: int
    y: int

def part1(data: list[str]):
    red_tiles = set()
    for line in data:
        x, y = map(int, line.split(","))
        red_tiles.add(Point(x, y))

    biggest_area = 0
    for point1, point2 in combinations(red_tiles, 2):
        # We need inclusive area
        rectangle_area = (abs(point2.x - point1.x) + 1) * (abs(point2.y - point1.y) + 1)
        biggest_area = max(biggest_area, rectangle_area)
            
    return biggest_area

class PolygonSolver:
    """ Solves the problem using Ray Casting and edge intersection checks. """
    def __init__(self, corners: list[Point]):
        self.corners = corners
        self.num_corners = len(corners)
        
        # Pre-calculate edges for intersection checks
        # Store as (x1, y1, x2, y2) tuples
        self.vertical_edges = []
        self.horizontal_edges = []
        
        for i in range(self.num_corners):
            p1 = corners[i]
            p2 = corners[(i + 1) % self.num_corners]
            
            if p1.x == p2.x: # Vertical
                # Store with y1 < y2 for simplified checking
                y_min, y_max = min(p1.y, p2.y), max(p1.y, p2.y)
                self.vertical_edges.append((p1.x, y_min, y_max))
            else: # Horizontal
                x_min, x_max = min(p1.x, p2.x), max(p1.x, p2.x)
                self.horizontal_edges.append((x_min, x_max, p1.y))
                
    def is_point_inside(self, px: float, py: float) -> bool:
        """ 
        Determines if a point is inside the polygon using Ray Casting. 
        Casts a horizontal ray to the right from (px, py).
        Odd intersections = Inside.
        """
        intersections = 0
        
        for vx, vy_min, vy_max in self.vertical_edges:
            # Check if ray crosses this vertical edge
            # Ray is y = py, x > px
            # Edge is x = vx, y in [vy_min, vy_max]
            
            # 1. Edge must be strictly to the right of the point
            if vx > px:
                # 2. Ray's Y must be within the edge's Y range
                # We use vy_min <= py < vy_max to avoid double counting vertices
                if vy_min <= py < vy_max:
                    intersections += 1
                    
        return intersections % 2 == 1

    def intersects_rect(self, r_min_x, r_min_y, r_max_x, r_max_y) -> bool:
        """ 
        Checks if any polygon edge strictly intersects the INTERIOR of the rectangle. 
        Touching the boundary is allowed.
        """
        # Check Vertical Edges
        for vx, vy_min, vy_max in self.vertical_edges:
            # Does vertical edge X fall strictly inside rect X range?
            if r_min_x < vx < r_max_x:
                # Does vertical edge Y range overlap strictly with rect Y range?
                # Overlap: max(A_min, B_min) < min(A_max, B_max)
                overlap_min = max(vy_min, r_min_y)
                overlap_max = min(vy_max, r_max_y)
                if overlap_min < overlap_max:
                    return True # Intersects
        
        # Check Horizontal Edges
        for hx_min, hx_max, hy in self.horizontal_edges:
            # Does horizontal edge Y fall strictly inside rect Y range?
            if r_min_y < hy < r_max_y:
                # Does horizontal edge X range overlap strictly with rect X range?
                overlap_min = max(hx_min, r_min_x)
                overlap_max = min(hx_max, r_max_x)
                if overlap_min < overlap_max:
                    return True
                    
        return False

def main():
    try:
        ac.write_puzzle_input_file(YEAR, DAY, locations)
        with open(locations.input_file, encoding="utf-8") as f:
            input_data = f.read().splitlines() # Most puzzles are multiline strings
            # input_data = f.read().strip() # Raw string
            
            logger.debug(dc.top_and_tail(input_data))
    except (ValueError, FileNotFoundError) as e:
        logger.error("Could not read input file: %s", e)
        return 1

    # Part 1 tests
    logger.setLevel(logging.DEBUG)
    sample_inputs = []
    sample_inputs.append(textwrap.dedent("""\
        7,1
        11,1
        11,7
        9,7
        9,5
        2,5
        2,3
        7,3"""))
    sample_answers = [50]
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
