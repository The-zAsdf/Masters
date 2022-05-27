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

iX = []
c = -1
for row in rows:
    if row[0] != c:
        iX.append(float(row[1]))
    c = row[0]

iY = []
c = -1
for row in rows:
    if row[0] != c:
        iY.append(float(row[2]))
    c = row[0]

plt.plot(np.asarray(iX), np.asarray(iY), label = 'Invariance')
plt.xlabel("Time")
plt.ylabel("Invariance")
plt.title("Invariance vs. time")
plt.legend()
plt.savefig('plots/iplot.png')
plt.show()

# print(iX)
# print(iY)
