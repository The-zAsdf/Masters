import csv
import numpy as np
import math
import matplotlib.pyplot as plt

file = open("data/G.txt", "r")

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
        dX.append(float(row[5]))
    c = row[0]

dY = []
for i in range(0,n):
    dY.append([])
    for j in range(0,n):
        dY[i].append([])
        for k in range(0,n):
            dY[i][j].append([])
            for l in range(0,n):
                dY[i][j][k].append([])

for i in range(0,n):
    for j in range(0,n):
        for k in range(0,n):
            for l in range(0,n):
                for row in rows:
                    if int(row[1]) == i and int(row[2]) == j and int(row[3]) == k and int(row[4]) == l:
                        dY[i][j][k][l].append(float(row[6]))

for i in range(0,n):
    for j in range(0,n):
        for k in range(0,n):
            for l in range(0,n):
                # if i == k and j == l and i != j:
                #     plt.plot(np.asarray(dX), np.asarray(dY[i][j][k][l]), label = 'G[{}][{}][{}][{}]'.format(i,j,k,l))
                #     plt.plot(np.asarray(dX), np.asarray(dY[j][i][k][l]), label = 'G[{}][{}][{}][{}]'.format(j,i,k,l))
                 plt.plot(np.asarray(dX), np.asarray(dY[i][j][k][l]), label = 'G[{}][{}][{}][{}]'.format(i,j,k,l))


plt.xlabel("Time")
plt.ylabel("G[i][j][k][l]")
plt.title("rank-4 terms vs. time")
plt.legend()
plt.savefig('plots/Gplot.png')
plt.show()

# print(dX)
# print(dY[0][1])
