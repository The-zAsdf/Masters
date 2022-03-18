#ifndef MATOPERATIONS_CUH
#define MATOPERATIONS_CUH

#include "PRBM.h"

__device__ static const float r_9 = 1.0 / 9.0;
__device__ static const float r_2_9 = 2.0 / 9.0;
__device__ static const float r_12 = 1.0 / 12.0;
__device__ static const float r_324 = 1.0 / 324.0;
__device__ static const float r_330 = 1.0 / 330.0;
__device__ static const float r_28 = 1.0 / 28.0;

__global__ void COPY(float **src, float **dest) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dest[i][j] = src[i][j];
    }
}

__global__ void RESET(float **mat) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        mat[i][j] = 0.0;
    }
}

__global__ void MULT(float **a, float **b, float **save) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ int N;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        for (int k = 0; k < N; k++) {
            save[i][j] += a[min(i,k)][abs(i-k)]*b[min(j,k)][abs(j-k)];
        }
    }
}

__global__ void ADD(float **src, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += src[i][j];
    }
}

__global__ void APPLYSLOPE(float **k, float **dst, float modifier) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += modifier*h*k[i][j];
    }
}

__global__ void SUMRK(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += h/6.0 *(k[0][i][j] + 2.0*k[1][i][j] +
                     2.0*k[2][i][j]+k[3][i][j]);
    }
}

__global__ void SUMDP(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += h*(0.0862*k[0][i][j] + 0.666*k[2][i][j] - 0.7857*k[3][i][j]
                     + 0.9570*k[4][i][j] + 0.0965*k[5][i][j] - 0.02*k[6][i][j]);
    }
}

__global__ void DPSLOPE1(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += r_2_9*h*k[0][i][j];
    }
}

__global__ void DPSLOPE2(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += r_12*h*(k[0][i][j] + 3.0*k[1][i][j]);
    }
}

__global__ void DPSLOPE3(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += r_324*h*(55.0*k[0][i][j] - 75.0*k[1][i][j]
                     + 200.0*k[2][i][j]);
    }
}

__global__ void DPSLOPE4(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += r_330*h*(83.0*k[0][i][j] - 195.0*k[1][i][j]
                     + 305.0*k[2][i][j] + 27.0*k[3][i][j]);
    }
}

__global__ void DPSLOPE5(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += r_28*h*(-19.0*k[0][i][j] + 63.0*k[1][i][j]
                     + 4.0*k[2][i][j] - 108.0*k[3][i][j] + 88.0*k[4][i][j]);
    }
}

__global__ void DPSLOPE6(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    extern __managed__ float h;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += 0.0025*h*(38.0*k[0][i][j] + 240.0*k[2][i][j]
                     - 243.0*k[3][i][j] + 330.0*k[4][i][j] + 35.0*k[5][i][j]);
    }
}

__global__ void DPERROR(float ***k, float **dst) {
    extern __managed__ int numElem;
    extern __managed__ ind **threadIndex;
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] = fabsf(0.0002*(44.0*k[0][i][j] - 330.0*k[2][i][j]
                    + 891.0*k[3][i][j] - 660.0*k[4][i][j] - 45.0*k[5][i][j]
                    + 100.0*k[6][i][j]));
    }
}

float findMax(float **mat, int *x, int *y) {
    extern __managed__ int N;
    float c = mat[0][0];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            if (c > mat[i][j]) {
                c = mat[i][j];
                *x = i;
                *y = j;
            }
        }
    }
    return c;
}

#endif
