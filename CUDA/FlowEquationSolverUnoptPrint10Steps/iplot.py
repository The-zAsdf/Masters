import csv
import numpy as np
import math
import matplotlib.pyplot as plt

file = open("data/i.txt", "r")

csvreader = csv.reader(file)
header = []
header = next(csvreader)
r = int(header[0])
n = int(header[1])

rows = []
for row in csvreader:
    rows.append(row)

dX = []
c = -1
for row in rows:
    if row[0] != c:
        dX.append(float(row[1]))
    c = row[0]

dY = []
for i in range(0,n):
    dY.append([])

for i in range(0,n):
    for j in range(0,n):
        for row in rows:
            if int(row[1]) == i and int(row[2]) == j:
                dY[i][j].append(float(row[4]))

for i in range(0, n):
    for j in range(0, n):
        if i != j:
            plt.plot(np.asarray(dX), np.asarray(dY[i][j]), label = 'h[{}][{}]'.format(i,j))

plt.xlabel("Time")
plt.ylabel("D[i][j]")
plt.title("NN interactions vs. time")
plt.legend()

plt.show()

# print(dX)
# print(dY[0][1])
