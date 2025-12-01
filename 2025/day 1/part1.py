dir_list = []
amount_list = []

with open("d1p1.txt", "r") as file:
  for line in file:
    line = line.strip()
    
    dir_list.append(line[0])
    amount_list.append(int(line[1:]))

current_amount = 50
zeros = 0

def loop(nr):
  return nr % 100

for i in range(len(amount_list)):
  if dir_list[i] == "R":
    local_amount = amount_list[i]
    current_amount += local_amount
    current_amount = loop(current_amount)
  if current_amount == 0:
    zeros += 1

print(zeros)
