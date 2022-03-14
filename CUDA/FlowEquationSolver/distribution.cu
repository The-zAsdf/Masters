#include <math.h>
#include <stdlib.h>
#include "erfinv.h"

__managed__ float *uniform;
__managed__ float **invGaus;
int s;
float j0;
float alpha;

// Gaussian inverse CDF
float gaussianICDF(float p, float d) {
    return my_erfinvf(2.0*p-1.0)*j0*sqrtf(2.0)/powf(d,alpha);
}

// Generate standard uniform distribution
void gererateSUD(int size, float j, float a) {
    cudaError_t err = cudaMallocManaged(&uniform, sizeof(float)*size);
    if (err != cudaSuccess) CUDAERROR(err);
    s = size;
    j0 = j;
    alpha = a;

    for (int i = 0; i < s; i++) {
        uniform[i] = (float) i/(float) (s-1);
    }
}

// Generate ICDF values
void generateICDF() {
    int r;
    cudaError_t err = cudaMallocManaged(&invGaus, sizeof(float*)*s);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < s; i++) {
        err = cudaMallocManaged(&invGaus[i], sizeof(float)*s);
        if (err != cudaSuccess) CUDAERROR(err);
        for (int j = 0; j < s; j++) {
            r = rand()%s;
            invGaus[i][j] = gaussianICDF(uniform[r], i);
        }
    }
}

float getSampleNumber(int i, int j) {
    int r = rand()%s;

    return invGaus[abs(i-j)][r];
}

void freeDistributions() {
    cudaFree(uniform);
    for (int i = 0; i < s; i++) {
        cudaFree(invGaus[i]);
    }
    cudaFree(invGaus);
}
