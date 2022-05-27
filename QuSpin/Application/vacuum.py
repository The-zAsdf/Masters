from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d
import numpy as np # generic math functions
import random
import csv
from math import sqrt
def genW():
    return (random.random()-0.5)*2.0*5.0

file = open('H2.txt')
csvreader = csv.reader(file)
rows = []
for row in csvreader:
        rows.append(row[0])
n = sqrt(len(rows))
q = []
for i in range(int(n)):
    q.append(float(rows[i*int(n)+i]))

L=2
J=1.0
D=0.025
basis=spinless_fermion_basis_1d(L=L, Nf = L//2)
print(basis)

h=[[5.0,i] for i in range(L-1)]
hop_left=[[+J,i,i+1] for i in range(L-1)]
hop_right=[[-J,i,i+1] for i in range(L-1)]
static=[['n',h],['+-',hop_left],['-+',hop_right]]
dynamic = []

no_checks = dict(check_symm=False)
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64, **no_checks)
print(H.toarray())

eval,estates=H.eigh()
print(eval)
