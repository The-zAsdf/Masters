#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "measureTime.h"
#include "IO.h"
#include "err.h"
#include "PRBM.h"
#include "distribution.h"

#define SAVES 100

__managed__ float **master;
__managed__ float **temp;
__managed__ float ***history;
__managed__ float ***kMat;
__managed__ ind **threadIndex;

__managed__ float W;
__managed__ float J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ float h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
int steps;                  // Total simulation steps


/* Matrix operations through CUDA
 */
__global__ void COPY(float **src, float **dest) {
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
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += src[i][j];
    }
}

__global__ void SUMKMAT(float ***k, float **dst) {
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        dst[i][j] += h/6.0 *(k[0][i][j]+2.0*k[1][i][j]+2.0*k[2][i][j]+k[3][i][j]);
    }
}

__global__ void generateMaster() {
    int i;
    int j;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        if (j >= N-i || id >= numElem || i >= N) {
            printf("id = %d (%d,%d) (%d)\n",id, i, j, N-i);
        }
        // Placeholder, please change ASAP.
        if (j == 0) {
            master[i][0] = (float)rand()/(float)(RAND_MAX/W);
        } else {
            master[i][j] = getSampleNumber(i,j);
        }
    }
}

void setVariables(struct Variables *v) {
    W = v->W;
    J = v->J;
    N = v->N[v->index];
    h = v->h;
    steps = v->steps;

    numElem = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            numElem++;
        }
    }
    determineThreadsAndBlocks();
}

size_t calculateBlocks(size_t threads) {
    size_t blocks = 1;
    for (size_t i = threads; i < numElem; i += threads) {
        if (blocks < threads*2 || threads == 1024) {
            blocks++;
        } else {
            return 0;
        }
    }
    return blocks;
}

/*  Keep the threads per block a multiple of 32 and the number of blocks as
 *  close as possible to the threads per block.
 *  Improves efficiency of CUDA calculations
 */
void determineThreadsAndBlocks() {
    size_t blocks;
    size_t threads = 0;
    do {
        threads += 16;
        blocks = calculateBlocks(threads);
    } while (blocks == 0 && threads < 1024);
    nob = blocks;
    tpb = threads;
}

void init() {
    time_t t;
    cudaError_t err;
    int count;

    srand((unsigned) time(&t));
    err = cudaMallocManaged(&master, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&temp, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&history, sizeof(float*)*SAVES);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&kMat, sizeof(float*)*4);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&threadIndex, sizeof(struct index *)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&threadIndex[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }

    // threadIndex
    count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            threadIndex[count]->x = i;
            threadIndex[count]->y = j;
            count++;
        }
    }

    // master and history
    for (int i = 0; i < N; i++){
        err = cudaMallocManaged(&master[i], sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i], sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
    }

    // init distribution
    generateSUD(N, J, 1.0);
    generateICDF();

    // initialize master values
    generateMaster<<<nob,tpb>>>();
    checkCudaSyncErr();

    // free distribution
    freeDistributions();

    // history
    for (int i = 0; i < SAVES; i++) {
        err = cudaMallocManaged(&history[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&history[i][j], sizeof(float)*(N-j));
            if (err != cudaSuccess) CUDAERROR(err);
        }
    }

    // kMat
    for (int i = 0; i < 4; i++) {
        err = cudaMallocManaged(&kMat[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i][j], sizeof(float)*(N-j));
            if (err != cudaSuccess) CUDAERROR(err);
        }
    }
}

void freeMem() {
    for (int i = 0; i < N; i++) {
        cudaFree(master[i]);
        cudaFree(temp[i]);
    }
    cudaFree(master);
    cudaFree(temp);

    for (int i = 0; i < SAVES; i++) {
        for (int j = 0; j < N; j++) {
            cudaFree(history[i][j]);
        }
        cudaFree(history[i]);
    }
    cudaFree(history);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < N; j++) {
            cudaFree(kMat[i][j]);
        }
        cudaFree(kMat[i]);
    }
    cudaFree(kMat);
}

__global__ void calckMat(float **kM, float **mat) {
    int i;
    int j;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;

        if (j == 0) {
            // funcH
            kM[i][0] = 0.0;
            for (int k = 0; k < N; k++) {
                kM[i][0] += powf(mat[min(i,k)][abs(i-k)], 2.0)*(mat[i][0]-mat[k][0]);
            }
        } else {
            // funcJ
            kM[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                kM[i][j] -= mat[min(i,k)][abs(i-k)]*mat[min(j,k)][abs(j-k)]*(2.0*mat[k][0]-mat[i][0]-mat[j][0]);
            }
            kM[i][j] -= mat[i][j]*powf(mat[i][0]-mat[j][0],2.0);
        }
    }
}

float adaptiveScaling(float q) {
    return q; // TBC; for later
}

void updateMat() {
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    calckMat<<<nob,tpb>>>(kMat[0], temp);
    checkCudaSyncErr();

    ADD<<<nob,tpb>>>(kMat[0], temp);
    checkCudaSyncErr();

    calckMat<<<nob,tpb>>>(kMat[1], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    ADD<<<nob,tpb>>>(kMat[1], temp);
    checkCudaSyncErr();

    calckMat<<<nob,tpb>>>(kMat[2], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    ADD<<<nob,tpb>>>(kMat[2], temp);
    checkCudaSyncErr();

    calckMat<<<nob,tpb>>>(kMat[3], temp);
    checkCudaSyncErr();

    SUMKMAT<<<nob,tpb>>>(kMat, master);
    checkCudaSyncErr();
}

double runPRBM(struct Variables *v) {
    printf("Setting variables:\n");
    setVariables(v);
    printf("Done\nInitializing:... ");
    init();
    printf("Done\nStarting simulation:\n");
    startTime();
    for (int s = 0; s < steps; s++) { updateMat(); }
    endTime();
    printf("Done\n");
    freeMem();
    return runTime();
}
