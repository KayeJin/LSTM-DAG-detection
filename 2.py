res = []
with open("res.txt", "r") as f:
    for line in f.readlines():
        if line[0:6] == 'benign' :
            res.append(line)

with open("benign.txt", "w") as f:
    for line in res:
        f.write(line)