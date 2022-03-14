#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include "erfinv.h"
#include "err.h"

__managed__ float *uniform;
__managed__ float **invGaus;
float alpha;
int numElem;
float J;
float W;

// Gaussian inverse CDF
float gaussianICDF(float p, float d) {
    return my_erfinvf(2.0*p-1.0)*J*sqrtf(2.0)/powf(d,alpha);
}

// Generate standard uniform distribution
void gererateSUD(float a, float j, float w, int num) {
    cudaError_t err = cudaMallocManaged(&uniform, sizeof(float)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    alpha = a;
    J = j;
    W = w;
    numElem = num;

    for (int i = 0; i < numElem; i++) {
        uniform[i] = (float) i/(float) (numElem-1);
    }
}

// Generate ICDF values
void generateICDF() {
    int r;
    cudaError_t err = cudaMallocManaged(&invGaus, sizeof(float*)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&invGaus[i], sizeof(float)*numElem);
        if (err != cudaSuccess) CUDAERROR(err);
        for (int j = 0; j < numElem; j++) {
            r = rand()%numElem;
            invGaus[i][j] = gaussianICDF(uniform[r], i);
        }
    }
}

void freeDistributions() {
    cudaFree(uniform);
    for (int i = 0; i < numElem; i++) {
        cudaFree(invGaus[i]);
    }
    cudaFree(invGaus);
}

__global__ void generateMaster(curandState_t* states, float** master) {
    int i;
    int j;
    int r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        if (j >= N-i || id >= numElem || i >= N) {
            printf("id = %d (%d,%d) (%d)\n",id, i, j, N-i);
        }
        if (j == 0) {
            master[i][0] = (float)curand(&states[id])/((float)RAND_MAX/(float)W);
        } else {
            r = curand(&states[id])%numElem;
            master[i][j] = invGaus[abs(i-j)][r];
        }
    }
}
