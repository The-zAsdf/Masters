import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def test(h,H2,D,N,n,r):
    THamSq=traceHamSq(h,H2,D,N,n,r)
    THamSS=traceHamSS(h,H2,D,N,n,r)
    val = []
    for i in range(0,r):
        val.append(float(THamSq[i]-THamSS[i]))
    return val

def traceHam(h,H2,D,N,n,r):
    TH2 = traceH2(h,N,n,r)
    TD = traceD(D,N,n,r)
    TH2D = traceH2D(D,h,N,n,r)
    val = []
    for i in range(0,r):
        val.append(float(TH2[i]+TD[i]))
    return val

def traceHamSS(h,H2,D,N,n,r):
    TH2 = traceH2(h,N,n,r)
    TD = traceD(D,N,n,r)
    TH2D = traceH2D(D,h,N,n,r)
    val = []
    for i in range(0,r):
        val.append(float(pow(TH2[i],2.0)+2.0*TH2[i]*TD[i]+pow(TD[i],2.0)))
    return val

def traceHamSq(h,H2,D,N,n,r):
    TH2D = traceH2D(D,h,N,n,r)
    TH2Sq = traceH2Sq(H2,h,N,n,r)
    TDSq = traceDSq(D,N,n,r)
    val = []
    for i in range(0,r):
        val.append(float(TH2Sq[i]+2.0*TH2D[i]+TDSq[i]))
    return val

def traceDSq(D,N,n,r):
    TD = traceD(D,N,n,r)
    Nbar = 1.0 - N
    val = []
    for i in range(0,r):
        q = 0.0
        for j in range(0,n):
            for k in range(0,n):
                q += 8.0*pow(N,2)*pow(Nbar,2)*pow(D[j][k][i],2.0)
                for l in range(0,n):
                    q += 16.0*pow(N,3.0)*Nbar*D[j][k][i]*D[k][j][i]
        val.append(float(q+pow(TD[i],2.0)))
    return val

def traceH2D(D,h,N,n,r):
    Th = traceH2(h,N,n,r)
    TD = traceD(D,N,n,r)
    Nbar = 1.0 - N
    val = []
    for i in range(0,r):
        q = 0.0
        for j in range(0,n):
            for k in range(0,n):
                q += h[j][i]*D[j][k][i]
        val.append(float(Th[i]*TD[i]-pow(N,2.0)*Nbar*q))
    return val

def traceH2Sq(H2,h,N,n,r):
    Th = traceH2(h,N,n,r)
    Nbar = 1.0 - N
    val = []
    for i in range(0,r):
        q = 0.0
        for j in range(0,n):
            for k in range(0,n):
                q += H2[j][k][i]*H2[k][j][i]
        val.append(float(pow(Th[i],2.0)+N*Nbar*q))
    return val

def traceD(D,N,n,r):
    val = []
    for i in range(0,r):
        q = 0.0
        for j in range(0,n):
            for k in range(0,n):
                q += D[j][k][i]
        val.append(float(-2.0*N*N*q))
    return val

def traceH2(h,N,n,r):
    val = []
    for i in range(0,r):
        q = 0.0
        for j in range(0,n):
            q += h[j][i]
        val.append(float(N*q))
    return val


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

def main():
    N = 0.5
    Nbar = 1.0 - N

    header,rows = readToArr("data/h.txt")
    r = int(header[0])
    n = int(header[1])
    t = getTimeR1(rows)
    h = getValR1(rows,n)

    header,rows = readToArr("data/H2mat.txt")
    H2 = getValR2(rows,n)

    header,rows = readToArr("data/D.txt")
    D = getValR2(rows,n)

    TH2 = traceH2(h,N,n,r)
    TD = traceD(D,N,n,r)
    TH2D = traceH2D(D,h,N,n,r)
    TH2Sq = traceH2Sq(H2,h,N,n,r)
    TDSq = traceDSq(D,N,n,r)
    THamSq = traceHamSq(h,H2,D,N,n,r)
    THamSS = traceHamSS(h,H2,D,N,n,r)
    Test = test(h,H2,D,N,n,r)
    THam = traceHam(h,H2,D,N,n,r)
    # plt.plot(np.asarray(t), np.asarray(TH2), label = '<h>')
    # plt.plot(np.asarray(t), np.asarray(TD), label = '<D>')
    # plt.plot(np.asarray(t), np.asarray(TH2D), label = '<hD>')
    # plt.plot(np.asarray(t), np.asarray(TH2Sq), label = '<h^2>')
    # plt.plot(np.asarray(t), np.asarray(TDSq), label = '<D^2>')
    plt.plot(np.asarray(t), np.asarray(THam), label = '<H>')
    plt.plot(np.asarray(t), np.asarray(THamSq), label = '<H^2>')
    plt.plot(np.asarray(t), np.asarray(THamSS), label = '<H>^2')
    plt.plot(np.asarray(t), np.asarray(Test), label = '<H^2>-<H>^2')
    plt.xlabel("Time")
    plt.ylabel("Invariance")
    plt.title("Invariance vs. time")
    plt.legend()
    # plt.savefig('plots/iplot.png')
    plt.xlim([0,t[len(t)-1]])
    plt.ylim([-5,10])
    plt.show()

if __name__ == "__main__":
    main()
