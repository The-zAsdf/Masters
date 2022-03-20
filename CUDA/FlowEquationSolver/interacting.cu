#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "measureTime.h"
#include "IO.h"
#include "err.h"
#include "PRBM.h"
#include "distribution.h"
#include "matOperations.cuh"

__managed__ float **master;
__managed__ float **temp;
__managed__ float ***kMat;
__managed__ float **invGaus;
__managed__ float *uniform;
__managed__ ind **threadIndex;

__managed__ float W;
__managed__ float J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ float h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
int steps;                  // Total simulation steps



__global__ void generateMaster(curandState_t* states) {
    int i;
    int j;
    int r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;
        if (j == 0) {
            master[i][0] = curand_uniform(&states[id])*W;
        } else {
            r = (int)(curand_uniform(&states[id])*((float) numElem));
            master[i][j] = invGaus[abs(i-j)][r];
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

__global__ void initStates(unsigned int seed, curandState_t* states) {
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void printStates(curandState_t* states) {
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        printf("(int)(curand_uniform(&states[%d])*((float) numElem)) = %d\n", id, (int)(curand_uniform(&states[id])*((float) numElem)));
    }
}

void init() {
    curandState_t* states;
    cudaError_t err;
    time_t t;
    int count;
    int r;

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

    // master and temp
    for (int i = 0; i < N; i++){
        err = cudaMallocManaged(&master[i], sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i], sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
    }

    // init distribution
    err = cudaMallocManaged(&uniform, sizeof(float)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGaus, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&invGaus[i], sizeof(float)*numElem);
        if (err != cudaSuccess) CUDAERROR(err);

        uniform[i] = (float) i/(float) (numElem-1);
    }

    // Setup cuRAND states + distribution
    cudaMallocManaged((void**) &states, numElem * sizeof(curandState_t));
    initStates<<<nob, tpb>>>((unsigned) time(&t), states);
    checkCudaSyncErr();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < numElem; j++) {
            r = rand()%numElem;
            invGaus[i][j] = gaussianICDF(uniform[r], (float) i+1, J, 1.0f);
        }
    }

    // initialize master values
    generateMaster<<<nob,tpb>>>(states);
    checkCudaSyncErr();

    // free distribution + cuRAND states
    cudaFree(states);
    for (int i = 0; i < numElem; i++) cudaFree(invGaus[i]);
    cudaFree(invGaus);
    cudaFree(uniform);

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

// Keep these functions in the same file as CALCSLOPE, or find a way to
// (efficiently) pass device code through kernal (i.e no memcopies)
__device__ void funcH(float **mat, float *q, int i, int j) {
    // float hi = mat[i][0];
    // *q = 0.0f;
    // for (int k = 0; k < N; k++) {
    //     if (i != k) {
    //         *q += powf(mat[min(i,k)][abs(i-k)], 2.0f)*(hi-mat[k][0]);
    //     }
    // }
    // *q *= 2.0f;
}

__device__ void funcJ(float **mat, float *q, int i, int j) {
    // float hi, hj;
    // int x = i;
    // int y = j+i;
    // *q = 0.0f;
    // hi = mat[x][0];
    // hj = mat[y][0];
    // for (int k = 0; k < N; k++) {
    //     if (x != k && y != k) {
    //         *q -= mat[min(x,k)][abs(x-k)]*mat[min(y,k)][abs(y-k)]*(2.0f*mat[k][0]-hi-hj);
    //     }
    // }
    // if (x != y) *q -= mat[i][j]*powf(hi-hj,2.0f);
}

__global__ void CALCSLOPE(float **kM, float **mat) {
    float hi, hj;
    int i, j, x, y;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->x;
        j = threadIndex[id]->y;

        if (j == 0) {
            // funcH
            funcH(mat, &kM[i][0], i, j);
        } else {
            // funcJ
            funcJ(mat, &kM[i][j], i, j);
        }
    }
}

void DP() {
    // Copy master into temp
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,prev);
    checkCudaSyncErr();

    // Calculate k[0]
    CALCSLOPE<<<nob,tpb>>>(kMat[0], temp);
    checkCudaSyncErr();

    APPLYSLOPE<<<nob,tpb>>>(kMat[0], temp, 0.5f);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[1], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    APPLYSLOPE<<<nob,tpb>>>(kMat[1], temp, 0.5f);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[2], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    APPLYSLOPE<<<nob,tpb>>>(kMat[2], temp, 1.0f);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[3], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(temp,master);
    checkCudaSyncErr();

    SUMDP<<<nob,tpb>>>(kMat, master);
    checkCudaSyncErr();
    // printf("Final master:\n");
    printMatrix(master, N);
    printf("\n");
}

double runPRBM(struct Variables *v) {
    printf("Setting variables:\n");
    setVariables(v);
    printf("Done\nInitializing:... ");
    init();
    printf("Done\nStarting simulation:\n");
    startTime();
    for (int s = 0; s < steps; s++) {
        updateMat();
    }

    endTime();
    printf("Done\n");
    freeMem();
    return runTime();
}
