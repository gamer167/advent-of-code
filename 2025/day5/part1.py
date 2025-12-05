import sys

def reader():
    return open(f"imput.txt", 'r').read().split("\n\n")

def main() -> None:
    inp_ranges, ids = reader()
    inp_ranges = sorted([*map(int, i.split("-"))] for i in inp_ranges.splitlines())
    ids = sorted([*map(int, ids.splitlines())])

    ranges = [inp_ranges[0]]
    for l, r in inp_ranges[1:]:
        if l <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], r)

        else:
            ranges.append([l, r])

    ans = 0
    i, p = 0, 0
    while i < len(ids) and p < len(ranges):
        if ids[i] < ranges[p][0]:
            i += 1

        elif ranges[p][0] <= ids[i] <= ranges[p][1]:
            i += 1
            ans += 1

        else:  # ranges[p][1] < ids[i]
            p += 1

    print(ans)


if __name__ == "__main__":
    main()
