import os
os.chdir(os.path.dirname(__file__))
from functools import cache


def reader():
  return open(f"input.txt", 'r').read().splitlines()


def part1():
  f = [list(map(int, l)) for l in reader()]
  A = 0
  for b in f:
    M = 0
    for i in range(len(b) - 1):
      M = max(M, b[i] * 10 + max(b[(i + 1):]))
    A += M
  print(A)

part1()
