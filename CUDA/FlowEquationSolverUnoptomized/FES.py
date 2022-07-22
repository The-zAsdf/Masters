import os
import subprocess as sub
import Qu
import numpy as np
import genh
import math as m
import pIO
import csv
from trace import calcInvars
from trace import relativeErrorInvar
from trace import calcRelativeErrorInvar
from itertools import combinations as comb
from tqdm import tqdm

# Check that the FES program finished as expected
def confirmExitCode(num,prog):
    if num != 0:
        raise Exception("ERROR: {} encountered an error ({})".format(prog,num))
        quit(-1)

# Calculate the relative error of eigenvalues
def relativeError(evalED,evalFE,L):
    evalED = sorted(evalED, key = float)
    evalFE = sorted(evalFE, key = float)
    ep = 0.0
    for i in range(len(evalED)):
        ep += abs((evalED[i]-evalFE[i])/evalED[i])
    return pow(0.5,L)*ep

def runFESarg(arg, fileName):
    print(f"W = {arg[0]}, J = {arg[1]}, D = {arg[2]}, S = {arg[4]}, L = {arg[5]}")
    p = sub.Popen(["./main",f"{arg[0]}",f"{arg[1]}",f"{arg[2]}",f"{arg[3]}",f"{arg[4]}",f"{arg[5]}",f"{arg[6]:g}",f"{arg[7]:g}"], stdout=sub.PIPE, universal_newlines=True)
    pbar = tqdm(total = arg[4], desc = 'FES progress', dynamic_ncols = True, bar_format = '{l_bar}{bar}| [{elapsed}]', leave = False)
    for line in p.stdout:
        output = line.replace('\n','')
        output = output.split(',')
        pbar.update(float(output[1]))
    pbar.close()
    p.wait()
    confirmExitCode(p.returncode, "main")
    print("Run time: {}\n\n".format(pIO.getRuntime()))

    Inv = calcRelativeErrorInvar()
    file = open(fileName, 'a')
    file.write(f"{arg[0]/arg[1]},{arg[2]/arg[1]},{arg[5]},{Inv},{m.log10(Inv)}\n")
    file.close()


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
    rows=pIO.getInput("comb.txt")
    W = pIO.readInputReturnEntries('W',rows)
    L = pIO.readInputReturnEntries('N',rows)
    J = pIO.readInput('J', rows)
    D = pIO.readInput('D', rows)
    e = pIO.readInput('e', rows)
    h = pIO.readInput('h', rows)
    S = pIO.readInput('S', rows)
    c = pIO.readInput('c', rows)

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
            b += D[j[0]][j[1]]
        q.append(b)
    return q

def main():
    dir = os.getcwd()

    # Read input (used for Quspin)
    rows=pIO.getInput("input.txt")
    W = pIO.readInput('W', rows)
    J = pIO.readInput('J', rows)
    D = pIO.readInput('D', rows)
    L = int(pIO.readInput('N', rows))
    e = pIO.readInput('e', rows)
    c = pIO.readInput('c', rows)
    h = pIO.readInput('h', rows)
    S = pIO.readInput('S', rows)
    # f = int(pIO.readInput('f', rows))
    f = 3

    # Create new inital h values for FES
    # genh.hBR(W,L)
    # genh.hSR(W,L)
    # genh.hGR(W,L)
    # Run FES
    # runFES(W,J,D,h,S,L,e,c)

    # Do exact diagonlization calculations through Quspin
    print("Calculcation ED:")
    basis,H = Qu.calcHam(L,J,D, Nf=f)
    evalED,estates = H.eigh()
    print("ED finished.\n")
    # Get FE h & D
    hFE = pIO.gethFES(L)
    DFE = pIO.getDFES(L)
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
