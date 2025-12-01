dir_list = []
amount_list = []

with open("d1p1.txt", "r") as file:
    for line in file:
        line = line.strip()

        dir_list.append(line[0])
        amount_list.append(int(line[1:]))

current_position = 50
zeros = 0

def hits_through_zero(current_position, distance, direction):
    # normalize the starting position into 0-99
    s = current_position % 100
    first_zero_hit_at = (((100 - s) if direction == "R" else s)) or 100 # if left evals to 0, it will be 100
    if distance < first_zero_hit_at:
        return 0
    return 1 + (distance - first_zero_hit_at) // 100 # always 1 hit from first_zero_hit_at + rest

for i in range(len(dir_list)):
    direction = dir_list[i]
    distance = amount_list[i]
    zeros += hits_through_zero(current_position, distance, direction)
    if direction == "R":
        current_position = (current_position + distance) % 100

print(zeros)
