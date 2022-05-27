from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spinless fermion basis
import numpy as np # generic math functions
import random
import csv
from math import sqrt
def genW():
    return (random.random()-0.5)*2.0*5.0
##### define model parameters #####
L=20 # system size
J=1.0 # hopping strength
D=0.025

file = open('H2.txt')
csvreader = csv.reader(file)
rows = []
for row in csvreader:
        rows.append(row[0])
n = sqrt(len(rows))
q = []
for i in range(int(n)):
    q.append(float(rows[i*int(n)+i]))

##### construct basis
basis=spinless_fermion_basis_1d(L=L, Nf=L//2)
print(basis)
# define site-coupling lists for operators
n_pot=[[genW(),i] for i in range(L)]
J_nn_right=[[-J,i,i+1] for i in range(L-1)]
J_nn_left=[[+J,i,i+1] for i in range(L-1)]
D_nn=[[D,i,i+1] for i in range(L-1)]
# static and dynamic lists
static=[["+-",J_nn_left],["-+",J_nn_right],["n",n_pot],["nn",D_nn]]
dynamic=[]
###### construct Hamiltonian
no_checks = dict(check_symm=False,check_herm=False)
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis, **no_checks)

print("\nHamiltonian:\n",H.toarray())
eval,estates=H.eigh()
print("\nEigenvalues:\n",eval)
