#ifndef MATOPERATIONS_CUH
#define MATOPERATIONS_CUH

#include "interacting.cuh"

__device__ __constant__ float r_9 = 1.0 / 9.0;
__device__ __constant__ float r_2_9 = 2.0 / 9.0;
__device__ __constant__ float r_12 = 1.0 / 12.0;
__device__ __constant__ float r_324 = 1.0 / 324.0;
__device__ __constant__ float r_330 = 1.0 / 330.0;
__device__ __constant__ float r_28 = 1.0 / 28.0;

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
        if (k == -1 && l == -1) mat->mat[i][j] = 0.0f;
        else                    mat->ten[i][j][k][l] = 0.0f;
    }
}

// TODO: Change to interacting basis for practice
// __global__ void MULT(float **a, float **b, float **save) {
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

        if (k == -1 && l == -1) {
            eta->mat[i][j] = src->mat[i][j]*(src->mat[i][i] - src->mat[j][j]);
        } else {
            eta->ten[i][j][k][l] = src->ten[i][j][k][l]*(src->mat[i][i]
                                 + src->mat[k][k] - src->mat[j][j]
                                 - src->mat[l][l]);
            if (j == l) {
                eta->ten[i][j][k][l] += 2*(src->ten[i][i][j][j]
                                      - src->ten[i][j][j][i])*src->mat[k][j];
            }
            if (k == l) {
                eta->ten[i][j][k][l] -= 2*(src->ten[i][i][j][j]
                                      - src->ten[i][j][j][i])*src->mat[k][k];
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

__global__ void SUMDP(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (l == -1 && k == -1) {
            dst->mat[i][j] += h*(0.0862*kM[0]->mat[i][j] + 0.666*kM[2]->mat[i][j]
                            - 0.7857*kM[3]->mat[i][j] + 0.9570*kM[4]->mat[i][j]
                            + 0.0965*kM[5]->mat[i][j] - 0.02*kM[6]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += h*(0.0862*kM[0]->ten[i][j][k][l]
                                    + 0.666*kM[2]->ten[i][j][k][l]
                                    - 0.7857*kM[3]->ten[i][j][k][l]
                                    + 0.9570*kM[4]->ten[i][j][k][l]
                                    + 0.0965*kM[5]->ten[i][j][k][l]
                                    - 0.02*kM[6]->ten[i][j][k][l]);
        }
    }
}

__global__ void DPSLOPE1(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += r_2_9*h*kM[0]->mat[i][j];
        } else {
            dst->ten[i][j][k][l] += r_2_9*h*kM[0]->ten[i][j][k][l];
        }
    }
}

__global__ void DPSLOPE2(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += r_12*h*(kM[0]->mat[i][j] + 3.0*kM[1]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += r_12*h*(kM[0]->ten[i][j][k][l]
                                  + 3.0*kM[1]->ten[i][j][k][l]);
        }
    }
}

__global__ void DPSLOPE3(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += r_324*h*(55.0*kM[0]->mat[i][j]
                            - 75.0*kM[1]->mat[i][j]
                            + 200.0*kM[2]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += r_324*h*(55.0*kM[0]->ten[i][j][k][l]
                            - 75.0*kM[1]->ten[i][j][k][l]
                            + 200.0*kM[2]->ten[i][j][k][l]);
        }
    }
}

__global__ void DPSLOPE4(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += r_330*h*(83.0*kM[0]->mat[i][j]
                            - 195.0*kM[1]->mat[i][j]
                            + 305.0*kM[2]->mat[i][j]
                            + 27.0*kM[3]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += r_330*h*(83.0*kM[0]->ten[i][j][k][l]
                            - 195.0*kM[1]->ten[i][j][k][l]
                            + 305.0*kM[2]->ten[i][j][k][l]
                            + 27.0*kM[3]->ten[i][j][k][l]);
        }
    }
}

__global__ void DPSLOPE5(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += r_28*h*(-19.0*kM[0]->mat[i][j]
                            + 63.0*kM[1]->mat[i][j]
                            + 4.0*kM[2]->mat[i][j]
                            - 108.0*kM[3]->mat[i][j]
                            + 88.0*kM[4]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += r_28*h*(-19.0*kM[0]->ten[i][j][k][l]
                            + 63.0*kM[1]->ten[i][j][k][l]
                            + 4.0*kM[2]->ten[i][j][k][l]
                            - 108.0*kM[3]->ten[i][j][k][l]
                            + 88.0*kM[4]->ten[i][j][k][l]);
        }
    }
}

__global__ void DPSLOPE6(struct floet **kM, struct floet *dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k == -1 && l == -1) {
            dst->mat[i][j] += 0.0025*h*(38.0*kM[0]->mat[i][j]
                            + 240.0*kM[2]->mat[i][j]
                            - 243.0*kM[3]->mat[i][j]
                            + 330.0*kM[4]->mat[i][j]
                            + 35.0*kM[5]->mat[i][j]);
        } else {
            dst->ten[i][j][k][l] += 0.0025*h*(38.0*kM[0]->ten[i][j][k][l]
                            + 240.0*kM[2]->ten[i][j][k][l]
                            - 243.0*kM[3]->ten[i][j][k][l]
                            + 330.0*kM[4]->ten[i][j][k][l]
                            + 35.0*kM[5]->ten[i][j][k][l]);
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
            dst->mat[i][j] = fabsf(0.0002*(44.0*kM[0]->mat[i][j]
                            - 330.0*kM[2]->mat[i][j]
                            + 891.0*kM[3]->mat[i][j]
                            - 660.0*kM[4]->mat[i][j]
                            - 45.0*kM[5]->mat[i][j]
                            + 100.0*kM[6]->mat[i][j]));
        } else {
            dst->ten[i][j][k][l] = fabsf(0.0002*(44.0*kM[0]->ten[i][j][k][l]
                            - 330.0*kM[2]->ten[i][j][k][l]
                            + 891.0*kM[3]->ten[i][j][k][l]
                            - 660.0*kM[4]->ten[i][j][k][l]
                            - 45.0*kM[5]->ten[i][j][k][l]
                            + 100.0*kM[6]->ten[i][j][k][l]));
        }
    }
}

#endif
