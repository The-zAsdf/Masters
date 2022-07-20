# Flowing Equation Solver

This Flowing Equation Solver uses CUDA combined with a Dormand-Prince integration scheme.

## Installation

If the FES is not running, try installing the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). After installing, restart your computer.

## Usage
Open a command prompt in the FES folder and execute the following:
```bash
C:\DirectoryToFES> main
```
The ```input.txt``` file contains all the necessary parameters for running the FES:
```text
W: Onsite potential
J: Interaction strength
D: Nearest-neighbour interactions
R: Leave at '1'. Will be used later for running multiple solvers
N: Length of lattice
h: Initial step size
S: Total runtime. Will terminate after solver has reached this point
e: Tolerance. Keep it around the initial value.
c: Cut-off range. Currently redundant. Keep it in, might cause error if omitted.
```
The FES will read ```input/H2.txt``` for the rank-2 part of the tensor. Undefined behavior if this is not included (Did not write any warnings/errors).

For a ```L=4``` system, the ```input/H2.txt``` file should read as follows:

```
(0,0)
(0,1)
(0,2)
(0,3)
(1,0)
.
.
.
(3,2)
(3,3)
```
where ```(i,j)``` corresponds to the matrix element of H(2)

## Output
There are 2 folders used for output
### Data
Outputs the rank-2 part and the diagonal part of the rank-4 tensor into ```H2.txt``` and ```D.txt``` respectively. It stores every element, at every step of the solver in the format:
```
timecount,i,j,timevalue,value
```

The runtime is also stored in ```time.txt```

### MathematicaInput
Stores the initial rank-2 and rank-4 tensor parts in ```H2_i.txt``` and ```H4_i.txt``` respectively. After the solver is finished, it stores the final rank-2 and rank-4 parts in ```H2_f.txt``` and ```H4_f.txt``` respectively.
