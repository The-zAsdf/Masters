import os
import subprocess as sub
import csv
import Qu
import numpy as np
import genh
import math as m
from trace import calcInvars
from trace import relativeErrorInvar
from trace import calcRelativeErrorInvar
from itertools import combinations as comb

# Read the runtime of FES from file
def getRuntime():
    file = open("data/time.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return rows[0][0]

# Check that the FES program finished as expected
def confirmExitCode(num,prog):
    if num != 0:
        raise Exception("ERROR: {} encountered an error ({})".format(prog,num))
        quit(-1)

# Interprets array of input
def readInputReturnEntries(char, input):
    for entry in input:
        if entry[0] == char:
            return entry
    raise Exception("ERROR: Input not found")
    quit(-1)

# Interprets the input
def readInput(char, input):
    for entry in input:
        if entry[0] == char:
            return float(entry[1])
    raise Exception("ERROR: Input not found")
    quit(-1)

# Reads input from file
def getInput(fileName):
    file = open(fileName, "r")
    csvreader = csv.reader(file, delimiter =' ')
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    return rows

def getDFES(L):
    # Read from file, get D(l = inf) entries
    file = open("data/D.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    c = -1
    index = -1
    count = 1
    # find the index of the last entries
    for i in range(1,len(rows)):
        if int(rows[i][0]) > c:
            c = int(rows[i][0])
            index = count
        count += 1
    # store D values in q array
    q = []
    for i in range(L):
        q.append([])
        for j in range(L):
            q[i].append(float(rows[index+i*L+j][4]))
    # Reverse the array
    # for i in range(L):
    #     q[i] = q[i][::-1]
    # q = q[::-1]
    return q

def gethFES(L):
    # Read from file, get h(l = inf) entries
    file = open("data/h.txt", "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    c = -1
    index = -1
    count = 1
    # find the index of the last entries
    for i in range(1,len(rows)):
        if int(rows[i][0]) > c:
            c = int(rows[i][0])
            index = count
        count += 1

    # store h values in q array
    q = []
    for i in range(L):
        q.append(float(rows[index+i][3]))

    # return reversed list, corresponds to ED evals
    # return q[::-1]
    return q

# Calculate the relative error of eigenvalues
def relativeError(evalED,evalFE,L):
    evalED = sorted(evalED, key = float)
    evalFE = sorted(evalFE, key = float)
    ep = 0.0
    for i in range(len(evalED)):
        ep += abs((evalED[i]-evalFE[i])/evalED[i])
    return pow(0.5,L)*ep

def runFES(W,J,D,h,S,L,e,c):
    # Run FES
    print("Running the FES:")
    print(f"  W = {W} (onsite interactions)")
    print("  J0 = {} (hopping amplitude)".format(J))
    print("  D0 = {} (NN interactions)".format(D))
    print("  L = {} (number of sites)".format(L))
    print(f"  e = {e:g} (accuracy)")
    print(f"  cutoff = {c:g} (accuracy)")
    # Usage: main <W> <J> <D> <h> <S> <N> <e> <c>
    p = sub.run(["./main",f"{W}",f"{J}",f"{D}",f"{h}",f"{S}",f"{L}",f"{e:g}",f"{c:g}"])
    # p = sub.run("main")
    confirmExitCode(p.returncode, "main")
    print("FES finished. Run time: {}\n".format(getRuntime()))

def invarVsL():
    #varying W,N only
    rows=getInput("comb.txt")
    W = readInputReturnEntries('W',rows)
    L = readInputReturnEntries('N',rows)
    J = readInput('J', rows)
    D = readInput('D', rows)
    e = readInput('e', rows)
    h = readInput('h', rows)
    S = readInput('S', rows)
    c = readInput('c', rows)

    q = []
    for i in range(1,len(W)):
        q.append([])
        for j in range(1,len(L)):
            runFES(float(W[i]),J,D,h,S,int(L[j]),e,c)
            q[i-1].append([float(W[i]),int(L[j]),calcRelativeErrorInvar()])

    return q

def hComb(h,L,Nf):
    ind = []
    for i in range(L):
        ind.append(i)

    c = comb(ind,Nf)
    q = []
    for i in list(c):
        b = 0.0
        for j in range(Nf):
            b += h[i[j]]
        q.append(b)
    return q

def DComb(D,L,Nf):
    ind = []
    for i in range(L):
        ind.append(i)

    c = comb(ind,Nf)
    q = []
    for i in list(c):
        b = 0.0
        csub = comb(i,2)
        for j in list(csub):
            b += D[j[0]][j[1]]
        q.append(b)
    return q

def evalComb(h,D,L,Nf):
    ind = []
    for i in range(L):
        ind.append(i)

    c = comb(ind,Nf)
    q = []
    for i in list(c):
        b = 0.0
        csub = comb(i,2)
        for j in range(Nf):
            b += h[i[j]]
        for j in list(csub):
            # THIS SHOULD BE -= DUE TO CONVENTION CHANGING
            b += D[j[0]][j[1]]
        q.append(b)
    return q

def main():
    dir = os.getcwd()

    # Read input (used for Quspin)
    rows=getInput("input.txt")
    W = readInput('W', rows)
    J = readInput('J', rows)
    D = readInput('D', rows)
    L = int(readInput('N', rows))
    e = readInput('e', rows)
    c = readInput('c', rows)
    h = readInput('h', rows)
    S = readInput('S', rows)
    # f = int(readInput('f', rows))
    f = 1

    # Create new inital h values for FES
    # genh.hBR(W,L)
    # genh.hSR(W,L)
    genh.hGR(W,L)
    # Run FES
    runFES(W,J,D,h,S,L,e,c)

    # Do exact diagonlization calculations through Quspin
    print("Calculcation ED:")
    basis,H = Qu.calcHam(L,J,D, Nf=f)
    evalED,estates = H.eigh()
    print("ED finished.\n")
    # Get FE h & D
    hFE = gethFES(L)
    DFE = getDFES(L)
    # Do combinatorics to obtain evals corresponding to ED states
    evalFEcomb = evalComb(hFE,DFE,L,f)

    # Output FE and ED stuff
    print("----------CLARITY OUTPUT----------")
    print(basis)
    print("\n--FE h (potential terms):")
    print(hFE)
    print("\n--FE D (NN interacting):")
    print(np.array(DFE))
    print("\n--ED eval:")
    # print(evalED)
    print(sorted(evalED, key = float))
    print(f"\n--FE eval (Nf = {f}):")
    # print(evalFEcomb)
    print(sorted(evalFEcomb,key = float))
    # print(f"\n--FE h (Nf = {f}):")
    # print(hComb(hFE,L,f))
    # print(f"\n--FE D (Nf = {f}):")
    # print(DComb(DFE,L,f))
    print("----------------------------------\n")
    print("Relative error (eval): {}\n".format(relativeError(evalED,evalFEcomb,L)))

    I0,Iinf = calcInvars(N=0.5)
    print("I(l=0)   = {}".format(I0))
    print("I(l=inf) = {}".format(Iinf))
    print("Relative error (invariance): dI = {}".format(relativeErrorInvar(I0,Iinf)))
    print("Relative error (invariance): log10(dI) = {}".format(m.log10(relativeErrorInvar(I0,Iinf))))
    # q = invarVsL()


if __name__ == "__main__":
    main()
