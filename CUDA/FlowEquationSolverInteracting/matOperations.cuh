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
__global__ void COPY(floet **src, floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            if (i >= N || j >= N || i < 0 || j < 0) {
                printf("Illegal access at (%d, %d, %d, %d) for thread %d\n", i, j, k, l, id);
            }
            dst[i]->d[j] = src[i]->d[j];
        } else if (l == -2) { // J_{ij}
            if (i >= N || j >= N-i || i < 0 || j < 0) {
                printf("Illegal access at (%d, %d, %d, %d) for thread %d\n", i, j, k, l, id);
            }
            dst[i]->j[j] = src[i]->j[j];
        } else if (l == -3) { // h_i
            if (i >= N || i < 0) {
                printf("Illegal access at (%d, %d, %d, %d) for thread %d\n", i, j, k, l, id);
            }
            dst[i]->h = src[i]->h;
        } else { // :G:_{ijkl}
            if (i >= N || j >= N-i || k >= N || l >= N || i < 0 || j < 0 || k < 0 || l < 0) {
                printf("Illegal access at (%d, %d, %d, %d) for thread %d\n", i, j, k, l, id);
            }
            dst[i]->g[j][k][l] = src[i]->g[j][k][l];
        }
    }
}

__global__ void RESET(floet **mat) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (l == -1) { // :D:_{ij}
            mat[i]->d[j] = 0.0f;
        } else if (l == -2) { // J_{ij}
            mat[i]->j[j] = 0.0f;
        } else if (l == -3) { // h_i
            mat[i]->h = 0.0f;
        } else { // :G:_{ijkl}
            mat[i]->g[j][k][l] = 0.0f;
        }
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

// // <<<nobHH,tpbHH>>>
// __global__ void MULTHH(floet **a, floet **b, floet **dst) {
//     extern __managed__ int N;
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//
//     for (int i = index; i < N; i += stride) dst[i]->h = a[i]->h * b[i]->h;
// }
//
// // <<<nobJJ, tpbJJ>>>
// __global__ void MULTJJ(floet **a, floet **b, floet **dst) {
//     extern __managed__ int triangN;
//     extern __managed__ int N;
//     extern __managed__ ind **threadIndexJJ;
//     float t;
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (id < triangN) {
//         threadIndexJJ[id]->i;
//         threadIndexJJ[id]->j;
//         t = 0.0f;
//         for (int k = 0; k < N; k++) {
//             t += a[min(i,k)]->j[abs(i-k)]*b[min(j,k)]->j[abs(j-k)];
//         }
//         dst[i]->j[j] = t;
//     }
// }
//
// // <<<nobJJ, tpbJJ>>>
// __global__ void MULTDD(floet **a, floet **b, floet **dst) {
//     extern __managed__ int triangN;
//     extern __managed__ int N;
//     extern __managed__ ind **threadIndexJJ;
//     float t;
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (id < triangN) {
//         threadIndexJJ[id]->i;
//         threadIndexJJ[id]->j;
//         t = 0.0f;
//         for (int k = 0; k < N; k++) {
//             t += a[min(i,k)]->j[abs(i-k)]*b[min(j,k)]->j[abs(j-k)];
//         }
//         dst[i]->j[j] = t;
//     }
// }
//
//
// __global__ void MULTGG(floet **a, floet **b, floet **dst) {
//     extern __managed__ int N;
//     extern __managed__ ind **threadIndexJJ;
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//
//     if (id < N^2) {
//         threadIndexDD[id]->i;
//         threadIndexDD[id]->j;
//         dst[i]->d[j] = 0.0f;
//         for (int k = 0; k < N; k++) {
//             dst[i]->d[j] += a[i]->d[k]*b[k]->d[j];
//         }
//     }
// }

__global__ void ADD(floet **src, floet **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += src[i]->d[j];
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += src[i]->j[j];
        } else if (l == -3) { // h_i
            dst[i]->h += src[i]->h;
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += src[i]->g[j][k][l];
        }
    }
}

__global__ void SUMDP(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += h*(0.0862*kM[0][i]->d[j] + 0.666*kM[2][i]->d[j]
                            - 0.7857*kM[3][i]->d[j] + 0.9570*kM[4][i]->d[j]
                            + 0.0965*kM[5][i]->d[j] - 0.02*kM[6][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += h*(0.0862*kM[0][i]->j[j] + 0.666*kM[2][i]->j[j]
                            - 0.7857*kM[3][i]->j[j] + 0.9570*kM[4][i]->j[j]
                            + 0.0965*kM[5][i]->j[j] - 0.02*kM[6][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += h*(0.0862*kM[0][i]->h + 0.666*kM[2][i]->h
                            - 0.7857*kM[3][i]->h + 0.9570*kM[4][i]->h
                            + 0.0965*kM[5][i]->h - 0.02*kM[6][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += h*(0.0862*kM[0][i]->g[j][k][l]
                            + 0.666*kM[2][i]->g[j][k][l]
                            - 0.7857*kM[3][i]->g[j][k][l]
                            + 0.9570*kM[4][i]->g[j][k][l]
                            + 0.0965*kM[5][i]->g[j][k][l]
                            - 0.02*kM[6][i]->g[j][k][l]);
        }
    }
}

__global__ void DPSLOPE1(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += r_2_9*h*kM[0][i]->d[j];
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += r_2_9*h*kM[0][i]->j[j];
        } else if (l == -3) { // h_i
            dst[i]->h += r_2_9*h*kM[0][i]->h;
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += r_2_9*h*kM[0][i]->g[j][k][l];
        }
    }
}

__global__ void DPSLOPE2(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += r_12*h*(kM[0][i]->d[j] + 3.0*kM[1][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += r_12*h*(kM[0][i]->j[j] + 3.0*kM[1][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += r_12*h*(kM[0][i]->h + 3.0*kM[1][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += r_12*h*(kM[0][i]->g[j][k][l]
                                + 3.0*kM[1][i]->g[j][k][l]);
        }
    }
}

__global__ void DPSLOPE3(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += r_324*h*(55.0*kM[0][i]->d[j] - 75.0*kM[1][i]->d[j]
                         + 200.0*kM[2][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += r_324*h*(55.0*kM[0][i]->j[j] - 75.0*kM[1][i]->j[j]
                         + 200.0*kM[2][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += r_324*h*(55.0*kM[0][i]->h - 75.0*kM[1][i]->h
                         + 200.0*kM[2][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += r_324*h*(55.0*kM[0][i]->g[j][k][l]
                                - 75.0*kM[1][i]->g[j][k][l]
                                + 200.0*kM[2][i]->g[j][k][l]);
        }
    }
}

__global__ void DPSLOPE4(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += r_330*h*(83.0*kM[0][i]->d[j] - 195.0*kM[1][i]->d[j]
                         + 305.0*kM[2][i]->d[j] + 27.0*kM[3][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += r_330*h*(83.0*kM[0][i]->j[j] - 195.0*kM[1][i]->j[j]
                         + 305.0*kM[2][i]->j[j] + 27.0*kM[3][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += r_330*h*(83.0*kM[0][i]->h - 195.0*kM[1][i]->h
                         + 305.0*kM[2][i]->h + 27.0*kM[3][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += r_330*h*(83.0*kM[0][i]->g[j][k][l]
                                - 195.0*kM[1][i]->g[j][k][l]
                                + 305.0*kM[2][i]->g[j][k][l]
                                + 27.0*kM[3][i]->g[j][k][l]);
        }
    }
}

__global__ void DPSLOPE5(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += r_28*h*(-19.0*kM[0][i]->d[j] + 63.0*kM[1][i]->d[j]
                         + 4.0*kM[2][i]->d[j] - 108.0*kM[3][i]->d[j]
                         + 88.0*kM[4][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += r_28*h*(-19.0*kM[0][i]->j[j] + 63.0*kM[1][i]->j[j]
                         + 4.0*kM[2][i]->j[j] - 108.0*kM[3][i]->j[j]
                         + 88.0*kM[4][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += r_28*h*(-19.0*kM[0][i]->h + 63.0*kM[1][i]->h
                         + 4.0*kM[2][i]->h - 108.0*kM[3][i]->h
                         + 88.0*kM[4][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += r_28*h*(-19.0*kM[0][i]->g[j][k][l]
                                + 63.0*kM[1][i]->g[j][k][l]
                                + 4.0*kM[2][i]->g[j][k][l]
                                - 108.0*kM[3][i]->g[j][k][l]
                                + 88.0*kM[4][i]->g[j][k][l]);
        }
    }
}

__global__ void DPSLOPE6(struct floet ***kM, struct floet **dst) {
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
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] += 0.0025*h*(38.0*kM[0][i]->d[j] + 240.0*kM[2][i]->d[j]
                          - 243.0*kM[3][i]->d[j] + 330.0*kM[4][i]->d[j]
                          + 35.0*kM[5][i]->d[j]);
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] += 0.0025*h*(38.0*kM[0][i]->j[j] + 240.0*kM[2][i]->j[j]
                          - 243.0*kM[3][i]->j[j] + 330.0*kM[4][i]->j[j]
                          + 35.0*kM[5][i]->j[j]);
        } else if (l == -3) { // h_i
            dst[i]->h += 0.0025*h*(38.0*kM[0][i]->h + 240.0*kM[2][i]->h
                          - 243.0*kM[3][i]->h + 330.0*kM[4][i]->h
                          + 35.0*kM[5][i]->h);
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] += 0.0025*h*(38.0*kM[0][i]->g[j][k][l]
                                + 240.0*kM[2][i]->g[j][k][l]
                                - 243.0*kM[3][i]->g[j][k][l]
                                + 330.0*kM[4][i]->g[j][k][l]
                                + 35.0*kM[5][i]->g[j][k][l]);
        }
    }
}

__global__ void DPERROR(floet ***kM, floet **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (l == -1) { // :D:_{ij}
            dst[i]->d[j] = fabsf(0.0002*(44.0*kM[0][i]->d[j]
                         - 330.0*kM[2][i]->d[j] + 891.0*kM[3][i]->d[j]
                         - 660.0*kM[4][i]->d[j] - 45.0*kM[5][i]->d[j]
                         + 100.0*kM[6][i]->d[j]));
        } else if (l == -2) { // J_{ij}
            dst[i]->j[j] = fabsf(0.0002*(44.0*kM[0][i]->j[j]
                         - 330.0*kM[2][i]->j[j] + 891.0*kM[3][i]->j[j]
                         - 660.0*kM[4][i]->j[j] - 45.0*kM[5][i]->j[j]
                         + 100.0*kM[6][i]->j[j]));
        } else if (l == -3) { // h_i
            dst[i]->h = fabsf(0.0002*(44.0*kM[0][i]->h
                         - 330.0*kM[2][i]->h + 891.0*kM[3][i]->h
                         - 660.0*kM[4][i]->h - 45.0*kM[5][i]->h
                         + 100.0*kM[6][i]->h));
        } else { // :G:_{ijkl}
            dst[i]->g[j][k][l] = fabsf(0.0002*(44.0*kM[0][i]->g[j][k][l]
                         - 330.0*kM[2][i]->g[j][k][l] + 891.0*kM[3][i]->g[j][k][l]
                         - 660.0*kM[4][i]->g[j][k][l] - 45.0*kM[5][i]->g[j][k][l]
                         + 100.0*kM[6][i]->g[j][k][l]));
        }
    }
}

#endif
