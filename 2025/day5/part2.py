"""
Author: Jose A. Romero
Edited by: Gamer 1.
Puzzle: Advent of Code (year=2025 ; day=5 ; task=2)
"""

import sys

def reader():
    return open(f"imput.txt", 'r').read().split("\n\n")


def main() -> None:
        inp_ranges, _ = reader()
        inp_ranges = sorted([*map(int, i.split("-"))] for i in inp_ranges.splitlines())

        ranges = [inp_ranges[0]]
        for l, r in inp_ranges[1:]:
            if l <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], r)

            else:
                ranges.append([l, r])

        print(sum(r - l + 1 for l, r in ranges))


if __name__ == "__main__":
        main()
