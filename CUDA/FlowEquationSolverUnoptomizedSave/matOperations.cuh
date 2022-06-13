#ifndef MATOPERATIONS_CUH
#define MATOPERATIONS_CUH

#include "interacting.cuh"

__device__ __constant__ double r_9 = 1.0 / 9.0;
__device__ __constant__ double r_2_9 = 2.0 / 9.0;
__device__ __constant__ double r_12 = 1.0 / 12.0;
__device__ __constant__ double r_324 = 1.0 / 324.0;
__device__ __constant__ double r_330 = 1.0 / 330.0;
__device__ __constant__ double r_28 = 1.0 / 28.0;

// Once memory management has been implemented for explicit host/device
// interactions, replace this function with an appropriate cudaMemCpy variation.
__global__ void COPY(struct floet *src, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ int N;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1) dst->mat[i][j] = src->mat[i][j];
        else                    dst->ten[i][j][k][l] = src->ten[i][j][k][l];
    }
}

__global__ void RESET(struct floet *mat) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) mat->mat[i][j] = 0.0;
        else                    mat->ten[i][j][k][l] = 0.0;
    }
}

__global__ void TEST(struct floet *mat) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            mat->mat[i][j] += 1.0;
            if (i != j) mat->mat[j][i] += 1.0;
        } else if (k != -1 && l != -1) {
            mat->ten[i][j][k][l] += 1.0;
            if ((i >= j && k > l) || (i == j && k < l)) mat->ten[l][k][j][i] += 1.0;
        }
    }
}



// TODO: Change to interacting basis for practice
// __global__ void MULT(double **a, double **b, double **save) {
//     extern __managed__ int numElem;
//     extern __managed__ ind **threadIndex;
//     extern __managed__ int N;
//     int i;
//     int j;
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (id < numElem) {
//         i = threadIndex[id]->x;
//         j = threadIndex[id]->y;
//         for (int k = 0; k < N; k++) {
//             save[i][j] += a[min(i,k)][abs(i-k)]*b[min(j,k)][abs(j-k)];
//         }
//     }
// }

__global__ void GENERATOR(struct floet *src, struct floet *eta) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ int N;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1 && i >= j) { // if i == j then eta_{ij} = 0
            eta->mat[i][j] = src->mat[i][j]*(src->mat[i][i] - src->mat[j][j]);
            if (i != j) eta->mat[j][i] = -eta->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                // [H0(2), H(4)]
                eta->ten[i][j][k][l] = src->ten[i][j][k][l]*(src->mat[i][i]
                                     + src->mat[j][j] - src->mat[k][k]
                                     - src->mat[l][l]);

                // [H0(4), H(2)]
                if (i == k) {
                    eta->ten[i][j][k][l] += src->mat[j][l]*(src->ten[i][j][i][j]
                                          - src->ten[l][i][l][i]);
                } if (i == l) {
                    eta->ten[i][j][k][l] += src->mat[j][k]*(src->ten[k][i][k][i]
                                          - src->ten[i][j][i][j]);
                } if (j == k) {
                    eta->ten[i][j][k][l] += src->mat[i][l]*(src->ten[l][j][l][j]
                                          - src->ten[i][j][i][j]);
                } if (j == l) {
                    eta->ten[i][j][k][l] += src->mat[i][k]*(src->ten[i][j][i][j]
                                          - src->ten[k][j][k][j]);
                }
                // [H0(4), H(4)]
                eta->ten[i][j][k][l] += 4.0*(src->ten[k][l][k][l]
                                      - src->ten[i][j][i][j])*src->ten[i][j][k][l];

                // Symmmetries
                eta->ten[j][i][k][l] = - eta->ten[i][j][k][l];
                eta->ten[i][j][l][k] = - eta->ten[i][j][k][l];
                eta->ten[j][i][l][k] = eta->ten[i][j][k][l];
            }
        }
    }
}

__global__ void ADD(struct floet *src, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) dst->mat[i][j] += src->mat[i][j];
        else                    dst->ten[i][j][k][l] += src->ten[i][j][k][l];
    }
}

__global__ void SUMDP(struct floet **kM, struct floet *dst, double ct) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    double num;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += h*(0.0862*kM[0]->mat[i][j] + 0.666*kM[2]->mat[i][j]
                            - 0.7857*kM[3]->mat[i][j] + 0.9570*kM[4]->mat[i][j]
                            + 0.0965*kM[5]->mat[i][j] - 0.02*kM[6]->mat[i][j]);
            if (fabs(dst->mat[i][j]) < ct) dst->mat[i][j] = 0.0;
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = h*(0.0862*kM[0]->ten[i][j][k][l]
                    + 0.666*kM[2]->ten[i][j][k][l]
                    - 0.7857*kM[3]->ten[i][j][k][l]
                    + 0.9570*kM[4]->ten[i][j][k][l]
                    + 0.0965*kM[5]->ten[i][j][k][l]
                    - 0.02*kM[6]->ten[i][j][k][l]);

                if (fabs(dst->ten[i][j][k][l] + num) < ct) {
                    dst->ten[i][j][k][l] = 0.0;
                    dst->ten[j][i][k][l] = 0.0;
                    dst->ten[i][j][l][k] = 0.0;
                    dst->ten[j][i][l][k] = 0.0;
                } else {
                    dst->ten[i][j][k][l] += num;
                    dst->ten[j][i][k][l] -= num;
                    dst->ten[i][j][l][k] -= num;
                    dst->ten[j][i][l][k] += num;
                }
                // dst->ten[i][j][k][l] += num;
                // dst->ten[j][i][k][l] -= num;
                // dst->ten[i][j][l][k] -= num;
                // dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE1(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    double num;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += r_2_9*h*kM[0]->mat[i][j];
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = r_2_9*h*kM[0]->ten[i][j][k][l];

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE2(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    double num;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += r_12*h*(kM[0]->mat[i][j] + 3.0*kM[1]->mat[i][j]);
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = r_12*h*(kM[0]->ten[i][j][k][l]
                    + 3.0*kM[1]->ten[i][j][k][l]);

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE3(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    double num;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += r_324*h*(55.0*kM[0]->mat[i][j]
                            - 75.0*kM[1]->mat[i][j]
                            + 200.0*kM[2]->mat[i][j]);
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = r_324*h*(55.0*kM[0]->ten[i][j][k][l]
                                      - 75.0*kM[1]->ten[i][j][k][l]
                                      + 200.0*kM[2]->ten[i][j][k][l]);

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE4(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    double num;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += r_330*h*(83.0*kM[0]->mat[i][j]
                            - 195.0*kM[1]->mat[i][j]
                            + 305.0*kM[2]->mat[i][j]
                            + 27.0*kM[3]->mat[i][j]);
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = r_330*h*(83.0*kM[0]->ten[i][j][k][l]
                                      - 195.0*kM[1]->ten[i][j][k][l]
                                      + 305.0*kM[2]->ten[i][j][k][l]
                                      + 27.0*kM[3]->ten[i][j][k][l]);

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE5(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    double num;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += r_28*h*(-19.0*kM[0]->mat[i][j]
                            + 63.0*kM[1]->mat[i][j]
                            + 4.0*kM[2]->mat[i][j]
                            - 108.0*kM[3]->mat[i][j]
                            + 88.0*kM[4]->mat[i][j]);
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = r_28*h*(-19.0*kM[0]->ten[i][j][k][l]
                                      + 63.0*kM[1]->ten[i][j][k][l]
                                      + 4.0*kM[2]->ten[i][j][k][l]
                                      - 108.0*kM[3]->ten[i][j][k][l]
                                      + 88.0*kM[4]->ten[i][j][k][l]);

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPSLOPE6(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ double h;
    double num;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1 && i >= j) {
            dst->mat[i][j] += 0.0025*h*(38.0*kM[0]->mat[i][j]
                            + 240.0*kM[2]->mat[i][j]
                            - 243.0*kM[3]->mat[i][j]
                            + 330.0*kM[4]->mat[i][j]
                            + 35.0*kM[5]->mat[i][j]);
            if (i != j) dst->mat[j][i] = dst->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                num = 0.0025*h*(38.0*kM[0]->ten[i][j][k][l]
                                      + 240.0*kM[2]->ten[i][j][k][l]
                                      - 243.0*kM[3]->ten[i][j][k][l]
                                      + 330.0*kM[4]->ten[i][j][k][l]
                                      + 35.0*kM[5]->ten[i][j][k][l]);

                dst->ten[i][j][k][l] += num;
                dst->ten[j][i][k][l] -= num;
                dst->ten[i][j][l][k] -= num;
                dst->ten[j][i][l][k] += num;
            }
        }
    }
}

__global__ void DPERROR(floet **kM, floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] = fabs(0.0002*(44.0*kM[0]->mat[i][j]
                            - 330.0*kM[2]->mat[i][j]
                            + 891.0*kM[3]->mat[i][j]
                            - 660.0*kM[4]->mat[i][j]
                            - 45.0*kM[5]->mat[i][j]
                            + 100.0*kM[6]->mat[i][j]));
        } else if (k != -1 && l != -1) {
            dst->ten[i][j][k][l] = fabs(0.0002*(44.0*kM[0]->ten[i][j][k][l]
                                 - 330.0*kM[2]->ten[i][j][k][l]
                                 + 891.0*kM[3]->ten[i][j][k][l]
                                 - 660.0*kM[4]->ten[i][j][k][l]
                                 - 45.0*kM[5]->ten[i][j][k][l]
                                 + 100.0*kM[6]->ten[i][j][k][l]));
        }
    }
}

int ISHERM(floet *q, int N) {
    int c = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(q->mat[i][j]-q->mat[j][i]) >= 0.000001) {
                printf("H check: (%d,%d) != (%d,%d)\n", i,j,j,i);
                printf(" (|%.4f - ",q->mat[i][j]);
                printf("%.4f| = ",q->mat[j][i]);
                printf("%.4f)\n", fabs(q->mat[i][j]-q->mat[j][i]));
                c = 0;
            }
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (fabs(q->ten[i][j][k][l]-q->ten[l][k][j][i]) >= 0.000001) {
                        printf("H check (%d,%d,%d,%d) != (%d,%d,%d,%d)",i,j,k,l,l,k,j,i);
                        printf(" (|%.4f - ",q->ten[i][j][k][l]);
                        printf("%.4f| = ",q->ten[l][k][j][i]);
                        printf("%.4f)\n", fabs(q->ten[i][j][k][l]-q->ten[l][k][j][i]));
                        c = 0;
                    }
                }
            }
        }
    }
    return c;
}

int ISANTIHERM(floet *q, int N) {
    int c = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(q->mat[i][j]+q->mat[j][i]) >= 0.000001) {
                printf("AH check: (%d,%d) != -(%d,%d)\n", i,j,j,i);
                printf(" (|%.4f + ",q->mat[i][j]);
                printf("%.4f| = ",q->mat[j][i]);
                printf("%.4f)\n", fabs(q->mat[i][j]+q->mat[j][i]));
                c = 0;
            }
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (fabs(q->ten[i][j][k][l]+q->ten[l][k][j][i]) >= 0.000001) {
                        printf("AH check (%d,%d,%d,%d) != -(%d,%d,%d,%d)",i,j,k,l,l,k,j,i);
                        printf(" (|%.4f + ",q->ten[i][j][k][l]);
                        printf("%.4f| = ",q->ten[l][k][j][i]);
                        printf("%.4f)\n", fabs(q->ten[i][j][k][l]+q->ten[l][k][j][i]));
                        c = 0;
                    }
                }
            }
        }
    }
    return c;
}


#endif
