from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spinless fermion basis
import numpy as np # generic math functions
import random
import csv
from math import sqrt

def readH(fileName):
    file = open(fileName, "r")
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)

    file.close()
    n = int(sqrt(len(rows)))
    q = []
    for i in range(n):
        q.append(float(rows[i*n+i][0]))
    return q

def calcHam(L,J,D,Nf=1):
    h = readH("in/H2.txt")

    #### construct basis
    basis=spinless_fermion_basis_1d(L=L, Nf=Nf)
    # define site-coupling lists for operators
    n_pot=[[h[i],i] for i in range(L)]
    J_nn_right=[[-J,i,i+1] for i in range(L-1)]
    J_nn_left=[[+J,i,i+1] for i in range(L-1)]
    D_nn=[[D,i,i+1] for i in range(L-1)]
    # static and dynamic lists
    static=[["+-",J_nn_left],["-+",J_nn_right],["n",n_pot],["nn",D_nn]]
    dynamic=[]
    ###### construct Hamiltonian
    no_checks = dict(check_symm=False,check_herm=False)
    H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis, **no_checks)

    eval,estates=H.eigh()
    # print(basis)
    # print(eval)
    return basis,H
