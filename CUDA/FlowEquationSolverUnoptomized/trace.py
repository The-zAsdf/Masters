import csv
import numpy as np
import math

def getValR2(rows,n):
    val = []
    for i in range(0,n):
        val.append([])
        for j in range(0,n):
            val[i].append([])
            for row in rows:
                if int(row[1]) == i and int(row[2]) == j:
                    val[i][j].append(float(row[4]))
    return val

def getValR1(rows,n):
    val = []
    for i in range(0,n):
        val.append([])
        for row in rows:
            if int(row[1]) == i:
                val[i].append(float(row[3]))
    return val

def getTimeR1(rows):
    t = []
    c = -1
    for row in rows:
        if row[0] != c:
            t.append(float(row[2]))
        c = row[0]
    return t

def readToArr(fileName):
    file = open(fileName, "r")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return (header,rows)

def Invar(h,H2,D,t,N,L):
    Nbar = 1.0 - N
    q = 0.0
    for i in range(L):
        for j in range(L):
            q += N * Nbar * H2[i][j][t]*H2[j][i][t]
            q += 8.0 * N**2.0 * Nbar**2.0 * D[i][j][t]**2.0
            q -= 8.0 * N**2.0 * Nbar * h[i][t]*D[i][j][t]

            for k in range(L):
                q += 16.0 * N**3.0 * Nbar * D[i][j][t]*D[j][k][t]
    return q


def calcInvars(N=0.5):
    Nbar = 1.0 - N

    header,rows = readToArr("data/h.txt")
    r = int(header[0])-1
    n = int(header[1])
    h = getValR1(rows,n)

    header,rows = readToArr("data/H2mat.txt")
    H2 = getValR2(rows,n)

    header,rows = readToArr("data/D.txt")
    D = getValR2(rows,n)

    I0 = Invar(h,H2,D,0,N,n)
    Iinf = Invar(h,H2,D,r,N,n)
    return (I0, Iinf)

def relativeErrorInvar(I0,Iinf):
    return abs((I0-Iinf)/I0)

def calcRelativeErrorInvar(N=0.5):
    I0, Iinf = calcInvars(N)
    return relativeErrorInvar(I0,Iinf)
