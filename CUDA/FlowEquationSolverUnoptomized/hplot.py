import csv
import numpy as np
import math
import matplotlib.pyplot as plt

file = open("data/h.txt", "r")

csvreader = csv.reader(file)
header = []
header = next(csvreader)
r = int(header[0])
n = int(header[1])

rows = []
for row in csvreader:
    rows.append(row)

hX = []
for i in range(0,r):
    hX.append(float(rows[i*n][2]))

hY = []
el = 0
for i in range(0,n):
    hY.append([])
    el += 1

for i in range(0,n):
    for row in rows:
        if int(row[1]) == i:
            hY[i].append(float(row[3]))

for q in range(0, el):
    plt.plot(np.asarray(hX), np.asarray(hY[q]), label = 'h[{}]'.format(q))

plt.xlabel("Time")
plt.ylabel("h[i]")
plt.title("interaction terms vs. time")
plt.legend()

plt.show()
