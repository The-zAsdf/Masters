#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "measureTime.h"
#include "IO.h"
#include "err.cuh"
#include "interacting.cuh"
#include "distribution.h"
#include "matOperations.cuh"

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

#define DEBUG

__managed__ floet *master;
__managed__ floet *prev;
__managed__ floet *temp;
__managed__ floet *gen;
__managed__ floet **kMat;
__managed__ float **invGaus;
__managed__ float *uniform;
__managed__ ind **threadIndex;
__managed__ ind **threadIndexJJ;

__managed__ float W;
__managed__ float J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ float h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
double l;                  // Total simulation steps



__global__ void generateMaster(curandState_t* states) {
    int i, j, k, l, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k != -1 && ((i == j && k == l) || (i == l && j == k))) { // init :D:_{ij}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*((float) numElem));
            if (i == j) master->ten[i][j][k][l] = invGaus[abs(i-k)][r];
            else if (i == l) master->ten[i][j][k][l] = invGaus[abs(i-j)][r];
        } else if (k != -1 && l != -1) { // init :G:_{ijkl}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*1000);
            master->ten[i][j][k][l] = invGaus[abs(i-j)][r];
        } else if (k == -1 && (i == j)) { // init h_i
            master->mat[i][j] = curand_uniform(&states[id])*W;
        } else if (k == -1 && (i != j)){ // init :J:_{ij}
            // PLACEHOLDER //
            master->mat[i][j] == 0.0f;
        }
    }
}

void setVariables(struct Variables *v) {
    cudaDeviceProp props;
    int deviceId;
    W = v->W;
    J = v->J;
    N = v->N[v->index];
    h = v->h;
    l = v->steps;

    numElem = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            numElem ++; // H(2)
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) numElem++; // H(4)
            }
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
    cudaDeviceProp props;
    int deviceId;
    size_t blocks;
    size_t threads = 0;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    do {
        threads += props.warpSize;
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

    // Allocating master. The references are locally close to each other and
    // the elements are allocated after references have been allocated. This
    // keeps the elements close to each other in memory
    #ifdef DEBUG
    printf("\n\tAllocating master: ");
    #endif
    err = cudaMallocManaged(&master, sizeof(struct floet*));
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&master->mat, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&master->ten, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&master->mat[i], sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master->ten[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&master->ten[i][j], sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&master->ten[i][j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating prev: ");
    #endif

    err = cudaMallocManaged(&prev, sizeof(struct floet*));
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&prev->mat, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&prev->ten, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&prev->mat[i], sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev->ten[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&prev->ten[i][j], sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&prev->ten[i][j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating temp: ");
    #endif
    err = cudaMallocManaged(&temp, sizeof(struct floet*));
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&temp->mat, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&temp->ten, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&temp->mat[i], sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp->ten[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&temp->ten[i][j], sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&temp->ten[i][j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating gen: ");
    #endif
    err = cudaMallocManaged(&gen, sizeof(struct floet*));
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&gen->mat, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&gen->ten, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&gen->mat[i], sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&gen->ten[i], sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&gen->ten[i][j], sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&gen->ten[i][j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating kMat: ");
    #endif
    err = cudaMallocManaged(&kMat, sizeof(struct floet**)*7);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < 7; i++) {
        err = cudaMallocManaged(&kMat[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&kMat[i]->mat, sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&kMat[i]->ten, sizeof(float*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i]->mat[j], sizeof(float)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i]->ten[j], sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&kMat[i]->ten[j][k], sizeof(float*)*N);
                if (err != cudaSuccess) CUDAERROR(err);

                for (int l = 0; l < N; l++) {
                    err = cudaMallocManaged(&kMat[i]->ten[j][k][l], sizeof(float)*N);
                    if (err != cudaSuccess) CUDAERROR(err);
                }
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating threadIndex: ");
    #endif
    err = cudaMallocManaged(&threadIndex, sizeof(struct index*)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&threadIndex[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tInitializing threadIndex: ");
    #endif

    // threadIndex. Each thread corresponds to a matrix element.
    count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            threadIndex[count]->i = i;
            threadIndex[count]->j = j;
            threadIndex[count]->k = -1;
            threadIndex[count]->l = -1;
            count++;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    // each :G: element has a thread
                    threadIndex[count]->i = i;
                    threadIndex[count]->j = j;
                    threadIndex[count]->k = k;
                    threadIndex[count]->l = l;
                    count++;
                }
            }
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating and initializing distributions: ");
    #endif
    // init distribution
    err = cudaMallocManaged(&uniform, sizeof(float)*1000);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGaus, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&invGaus[i], sizeof(float)*1000);
        if (err != cudaSuccess) CUDAERROR(err);

        uniform[i] = (float) i/(float) (1000-1);
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating and initializing states: ");
    #endif

    // Setup cuRAND states + distribution
    cudaMallocManaged((void**) &states, 1000 * sizeof(curandState_t));
    initStates<<<nob, tpb>>>((unsigned) time(&t), states);
    checkCudaSyncErr();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 1000; j++) {
            r = rand()%1000;
            invGaus[i][j] = gaussianICDF(uniform[r], (float) i+1, J, 1.0f);
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tgenerateMaster: ");
    #endif

    // initialize master values
    generateMaster<<<nob,tpb>>>(states);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");

    printf("\tfreeing states and distributions: ");
    #endif

    // free distribution + cuRAND states
    cudaFree(states);
    cudaFree(invGaus);
    cudaFree(uniform);
    #ifdef DEBUG
    printf("Done\n");
    #endif

}

void freeMem() {
    for (int i = 0; i < N; i++) {
        cudaFree(master->mat[i]);
        cudaFree(prev->mat[i]);
        cudaFree(temp->mat[i]);
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                cudaFree(master->ten[i][j][k]);
                cudaFree(prev->ten[i][j][k]);
                cudaFree(temp->ten[i][j][k]);
            }
            cudaFree(master->ten[i][j]);
            cudaFree(prev->ten[i][j]);
            cudaFree(temp->ten[i][j]);
        }
        cudaFree(master->ten[i]);
        cudaFree(prev->ten[i]);
        cudaFree(temp->ten[i]);
    }
    cudaFree(master);
    cudaFree(temp);
    cudaFree(prev);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < N; j++) {
            cudaFree(kMat[i]->mat[j]);

            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    cudaFree(kMat[i]->ten[j][k][l]);
                }
                cudaFree(kMat[i]->ten[j][k]);
            }
            cudaFree(kMat[i]->ten[j]);
        }
        cudaFree(kMat[i]);
    }
    cudaFree(kMat);
}

// Keep these functions in the same file as CALCSLOPE, or find a way to
// (efficiently) pass device code through kernal (i.e no memcopies)
__device__ void funcH(struct floet *mat, float *q, int i) {

}

__device__ void funcJ(struct floet *mat, float *q, int i, int j) {

}

__device__ void funcD(struct floet *mat, float *q, int i, int j, int k, int l) {

}

__device__ void funcG(struct floet *mat, float *q, int i, int j, int k, int l) {

}

// Optimize this for MASSIVE performance increase (optimized matrix multiplication)
__global__ void CALCSLOPE(struct floet *kM, struct floet *mat) {
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k != -1 && ((i == j && k == l) || (i == l && j == k))) { // funcD
            funcD(mat, &kM->ten[i][j][k][l], i, j, k, l);
        } else if (k != -1 && l != -1) { // funcG
            funcG(mat, &kM->ten[i][j][k][l], i, j, k, l);
        } else if (k == -1 && (i == j)) { // funcH
            funcH(mat, &kM->mat[i][j], i);
        } else if (k == -1 && (i != j)) { // funcJ
            funcJ(mat, &kM->mat[i][j], i, j);
        }
    }
}

void DP () {
    #ifdef DEBUG
    printf("\t: ");
    #endif
    GENERATOR<<<nob, tpb>>>(master, gen);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,prev);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[0], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE1: ");
    #endif
    DPSLOPE1<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[1], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE2: ");
    #endif
    DPSLOPE2<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[2], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE3: ");
    #endif
    DPSLOPE3<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[3], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE4: ");
    #endif
    DPSLOPE4<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[4], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE5: ");
    #endif
    DPSLOPE5<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[5], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCopy: ");
    #endif
    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPSLOPE6: ");
    #endif
    DPSLOPE6<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tCalcslope: ");
    #endif
    CALCSLOPE<<<nob,tpb>>>(kMat[6], temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tSumDP: ");
    #endif
    SUMDP<<<nob,tpb>>>(kMat, master);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif

    #ifdef DEBUG
    printf("\tDPErr: ");
    #endif
    DPERROR<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
    #ifdef DEBUG
    printf("Done\n");
    #endif
}

void embeddedDP () {
    double s = 0.0;
    double scale;
    float err, t;
    double qq;
    double tol = 0.001/l;
    int last_interval = 0;
    int i, x, y, z, q;
    while (s < l) {
        scale = 1.0;
        for (i = 0; i < ATTEMPTS; i++) {
            #ifdef DEBUG
                printf("Starting DP:\n");
            #endif
            DP();
            #ifdef DEBUG
                printf("DP done\n");
            #endif
            err = findMax(temp, &x, &y, &z, &q);
            if (roundf(err) == err && roundf(err) == 0) {
                scale = MAX_SCALE_FACTOR;
                break;
            }
            t = readFloet(prev,x,y,z,q);
            if (roundf(t) == t && roundf(t) == 0.0f) {
                qq = tol;
            } else {
                qq = fabsf(t);
            }
            scale = 0.8 * sqrt( sqrt ( tol * qq /  (double) err ) );
            scale = min( max(scale,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);
            if ((double) err < (tol * qq)) break;
            h *= (float) scale;
            if (s + (double) h > l) h = (float)l - (float)s;
            else if (s + (double)h + 0.5*(double)h > l) h = 0.5f * h;
            COPY<<<nob,tpb>>>(prev,master);
            checkCudaSyncErr();

        }
        if ( i >= ATTEMPTS ) { printf("tolerance too small?\n"); exit(-2); }
        printf("s = %.4f, h = %.4f, scale = %.4f\n", s, h, scale);
        s += h;
        h *= scale;
        if ( last_interval ) break;
        if (s + (double) h > l) { last_interval = 1; h = (float) l - (float) s; }
        else if (s + h + 0.5*h > l) h = 0.5 * h;
        // printMatrix(master, N);
        // printf("\n");
    }
}

double runPRBM(struct Variables *v) {
    printf("Setting variables:\n");
    setVariables(v);
    printf("Done\nInitializing:... ");
    init();
    printf("Done\nStarting simulation:\n");
    startTime();
    embeddedDP();

    endTime();
    printf("Done\n");
    freeMem();
    return runTime();
}

float readFloet(struct floet *mat, int i, int j, int k, int l) {
    if (k == -1 && l == -1) return mat->mat[i][j];
    else                    return mat->ten[i][j][k][l];
}

////////////////////////////////////////////////////////////////////////////////
// Specific matrix based operations for the interacting model                 //
////////////////////////////////////////////////////////////////////////////////

float findMax(struct floet *mat, int *x, int *y, int *z, int *q) {
    float c = -1.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (mat->mat[i][j] > c) {
                c = mat->mat[i][j];
                *x = i;
                *y = j;
                *z = -1;
                *q = -1;
            }

            for (int k = 0; k < N; k++) {
                for (int m = 0; m < N; m++) {
                    if (mat->ten[i][j][k][m] > c) {
                        c = mat->ten[i][j][k][m];
                        *x = i;
                        *y = j;
                        *z = k;
                        *q = m;
                    }
                }
            }
        }
    }
    return c;
}
