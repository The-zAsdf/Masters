#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <complex.h>
#include <curand.h>
#include <curand_kernel.h>
#include "measureTime.h"
#include "IO.h"
#include "err.cuh"
#include "interacting.cuh"
#include "distribution.h"
#include "matOperations.cuh"

// #define DEBUG

#define NUMRECORDS 4000
#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 1.5
#define _USE_MATH_DEFINES
#include <math.h>

#define BRONZE_RATIO (3.0+sqrt(13.0))/2
#define SILVER_RATIO (1.0+sqrt(2.0))
#define GOLDEN_RATIO (1.0+sqrt(5.0))/2

__managed__ floet *master;
__managed__ floet *prev;
__managed__ floet *temp;
__managed__ floet *gen;
__managed__ floet **kMat;
__managed__ double **invGausJ;
__managed__ double **invGausD;
__managed__ double *uniform;
__managed__ double *hRead;
__managed__ ind **threadIndex;
__managed__ ind **threadIndexOpt;

__managed__ double W;
__managed__ double D;
__managed__ double ct;
__managed__ double etol;
__managed__ double J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ int numElemOpt;    // Number of elements
__managed__ double h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t tpbOpt;     // Threads per block
__managed__ size_t nob;     // Number of blocks
__managed__ size_t nobOpt;     // Number of blocks
double l;                   // Total simulation steps

floardH **hRecord;
floardD **dRecord;
floardD **H2Record;
int r;
int count;

__global__ void generateMaster(curandState_t* states) {
    int i, j, k, l, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    // double phi;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        // if (k != -1 && (i == j && k == l) && i > k) { // init :D:_{ij}
        //     // PLACEHOLDER //
        //     r = (int)(curand_uniform(&states[id])*1001 - 1);
        //     if ( i == k + 1 || k == i + 1) {
        //         master->ten[i][i][k][k] = J*0.05/2.0;
        //         master->ten[k][k][i][i] = master->ten[i][i][k][k];
        //         master->ten[i][k][k][i] = -J*0.05/2.0;
        //         master->ten[k][i][i][k] = master->ten[i][k][k][i];
        //     }
        // } else
        if (k == -1 && l == -1 && (i == j)) { // init h_i
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            master->mat[i][j] = (uniform[r]-0.5)*2.0*W;
        } else if (k == -1 && l == -1 && (i > j)){ // init :J:_{ij}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            if (i == j + 1 || j == i + 1) {
                master->mat[i][j] = J;
                master->mat[j][i] = J;
            }
        }
    }
}

__global__ void generateMasterH2(curandState_t* states) {
    int i, j, k, l;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    // double phi;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1 && (i == j)) { // init h_i
            master->mat[i][j] = W*hRead[i];
        } else if (k == -1 && l == -1 && (i > j)){ // init :J:_{ij}
            if (i == j + 1 || j == i + 1) {
                master->mat[i][j] = J;
                master->mat[j][i] = J;
            }
        }
    }
}

__global__ void generateMasterH4(curandState_t* states) {
    int i, j, k, l;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    // double phi;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k != -1 && l != -1) { // init :D:_{ij}
            // PLACEHOLDER //
            if ( i == k + 1 || k == i + 1 || i == l + 1 || l == i + 1) {
                if (i == k && j == l) {
                    master->ten[i][j][k][l] = -D/4.0;
                }
                if (i == l && j == k) {
                    master->ten[i][j][k][l] = D/4.0;
                }
            }
        }
    }
}

void inputMaster() {
    inputH2("H2", N, master->mat);
}

void setVariables(struct Variables *v) {
    // cudaDeviceProp props;
    // int deviceId;
    W = v->W;
    J = v->J;
    N = v->N[v->index];
    h = v->h;
    l = v->steps;
    etol = v->etol;
    D = v->D;
    ct = v->cutoff;

    numElem = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            numElem ++; // H(2)
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) numElem++; // H(4)
            }
        }
    }

    numElemOpt = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            numElemOpt++;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    numElemOpt++;
                }
            }
        }
    }

    determineThreadsAndBlocks(&nob, &tpb, numElem);
    determineThreadsAndBlocks(&nobOpt, &tpbOpt, numElemOpt);
    // printf("numElem = %d, nob = %zd, tpb = %zd\n", numElem, nob, tpb);
    // printf("numElemOpt = %d, nobOpt = %zd, tpbOpt = %zd\n", numElemOpt, nobOpt, tpbOpt);
}

size_t calculateBlocks(size_t threads, int num) {
    size_t blocks = 1;
    for (size_t i = threads; i < num; i += threads) {
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
void determineThreadsAndBlocks(size_t *b, size_t *t, int num) {
    cudaDeviceProp props;
    int deviceId;
    size_t blocks;
    size_t threads = 0;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    do {
        threads += props.warpSize;
        blocks = calculateBlocks(threads, num);
    } while (blocks == 0 && threads < 1024);
    *b = blocks;
    *t = threads;
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
        printf("(int)(curand_uniform(&states[%d])*((double) numElem)) = %d\n", id, (int)(curand_uniform(&states[id])*((double) numElem)));
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

    err = cudaMallocManaged(&master->mat, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&master->ten, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&master->mat[i], sizeof(double)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master->ten[i], sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&master->ten[i][j], sizeof(double*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&master->ten[i][j][k], sizeof(double)*N);
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

    err = cudaMallocManaged(&prev->mat, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&prev->ten, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&prev->mat[i], sizeof(double)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev->ten[i], sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&prev->ten[i][j], sizeof(double*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&prev->ten[i][j][k], sizeof(double)*N);
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

    err = cudaMallocManaged(&temp->mat, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&temp->ten, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&temp->mat[i], sizeof(double)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp->ten[i], sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&temp->ten[i][j], sizeof(double*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&temp->ten[i][j][k], sizeof(double)*N);
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

    err = cudaMallocManaged(&gen->mat, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&gen->ten, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&gen->mat[i], sizeof(double)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&gen->ten[i], sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&gen->ten[i][j], sizeof(double*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&gen->ten[i][j][k], sizeof(double)*N);
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

        err = cudaMallocManaged(&kMat[i]->mat, sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&kMat[i]->ten, sizeof(double*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i]->mat[j], sizeof(double)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i]->ten[j], sizeof(double*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&kMat[i]->ten[j][k], sizeof(double*)*N);
                if (err != cudaSuccess) CUDAERROR(err);

                for (int l = 0; l < N; l++) {
                    err = cudaMallocManaged(&kMat[i]->ten[j][k][l], sizeof(double)*N);
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

    err = cudaMallocManaged(&threadIndexOpt, sizeof(struct index*)*numElemOpt);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&threadIndex[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }
    for (int i = 0; i < numElemOpt; i++) {
        err = cudaMallocManaged(&threadIndexOpt[i], sizeof(struct index));
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
                    // each rank 4 element has a thread
                    threadIndex[count]->i = i;
                    threadIndex[count]->j = j;
                    threadIndex[count]->k = k;
                    threadIndex[count]->l = l;
                    count++;
                }
            }
        }
    }

    count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            threadIndexOpt[count]->i = i;
            threadIndexOpt[count]->j = j;
            threadIndexOpt[count]->k = -1;
            threadIndexOpt[count]->l = -1;
            count++;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = k+1; l < N; l++) {
                    // each rank 4 element has a thread
                    threadIndexOpt[count]->i = i;
                    threadIndexOpt[count]->j = j;
                    threadIndexOpt[count]->k = k;
                    threadIndexOpt[count]->l = l;
                    count++;
                }
            }
        }
    }

    // for (int i = 0; i < count; i++) {
    //     printf("ThreadOpt[%d] -> (%d,%d,%d,%d)\n", i,threadIndexOpt[i]->i
    //                                                 ,threadIndexOpt[i]->j
    //                                                 ,threadIndexOpt[i]->k
    //                                                 ,threadIndexOpt[i]->l);
    // }
    // exit(0);
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating and initializing distributions: ");
    #endif
    // init distribution
    err = cudaMallocManaged(&uniform, sizeof(double)*1000);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGausJ, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGausD, sizeof(double*)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&invGausJ[i], sizeof(double)*1000);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&invGausD[i], sizeof(double)*1000);
        if (err != cudaSuccess) CUDAERROR(err);
    }
    for (int i = 0; i < 1000; i++) uniform[i] = ((double) i)/((double) (1000-1));
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
            invGausJ[i][j] = gaussianICDF(uniform[r], (double) i+1, J, 1.0);
            r = rand()%1000;
            invGausD[i][j] = gaussianICDF(uniform[r], (double) i+1, J*0.1, 1.0);
        }
    }
    #ifdef DEBUG
    printf("Done\n");

    printf("\tgenerateMaster: ");
    #endif

    // initialize master values
    err = cudaMallocManaged(&hRead, sizeof(double)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    inputh("h", N, hRead);
    RESET<<<nob,tpb>>>(master);
    checkCudaSyncErr();
    generateMasterH4<<<nob,tpb>>>(states);
    checkCudaSyncErr();
    generateMasterH2<<<nob,tpb>>>(states);
    checkCudaSyncErr();
    // inputMaster();
    #ifdef DEBUG
    printf("Done\n");

    printf("\tfreeing states and distributions: ");
    #endif

    // free distribution + cuRAND states
    cudaFree(states);
    cudaFree(invGausJ);
    cudaFree(invGausD);
    cudaFree(uniform);
    cudaFree(hRead);
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating record objects:");
    #endif

    hRecord = (struct floardH**)malloc(sizeof(struct floardH*)*NUMRECORDS);
    dRecord = (struct floardD**)malloc(sizeof(struct floardD*)*NUMRECORDS);
    H2Record = (struct floardD**)malloc(sizeof(struct floardD*)*NUMRECORDS);

    for (int i = 0; i < NUMRECORDS; i++) {
        hRecord[i] = (struct floardH *)malloc(sizeof(struct floardH));
        hRecord[i]->h = (double*) malloc(sizeof(double)*N);
        dRecord[i] = (struct floardD *)malloc(sizeof(struct floardD));
        H2Record[i] = (struct floardD *)malloc(sizeof(struct floardD));
        dRecord[i]->D = (double**) malloc(sizeof(double*)*N);
        H2Record[i]->D = (double**) malloc(sizeof(double*)*N);
        for (int j = 0; j < N; j++) {
            dRecord[i]->D[j] = (double*)malloc(sizeof(double)*N);
            H2Record[i]->D[j] = (double*)malloc(sizeof(double)*N);
        }
    }

    #ifdef DEBUG
    printf("Done\n");
    #endif
}

// Optimize this for MASSIVE performance increase (optimized matrix multiplication)
__global__ void CALCSLOPE(struct floet *kM, struct floet *mat) {
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElemOpt) {
        i = threadIndexOpt[id]->i;
        j = threadIndexOpt[id]->j;
        k = threadIndexOpt[id]->k;
        l = threadIndexOpt[id]->l;

        if (k == -1 && l == -1 && i <= j) {
            kM->mat[i][j] = 0.0;
            // [eta(2),H(2)]
            for (int q = 0; q < N; q++) {
                kM->mat[i][j] += gen->mat[i][q]*mat->mat[q][j]
                               - mat->mat[i][q]*gen->mat[q][j];
            }
            if (i != j) kM->mat[j][i] = kM->mat[i][j];
        } else if (k != -1 && l != -1) {
            if (i < j && k < l) {
                kM->ten[i][j][k][l] = 0.0;
                // [eta(2),H(4)] + [eta(4),H(2)]
                for (int q = 0; q < N; q++) {
                     kM->ten[i][j][k][l] += mat->ten[q][j][k][l]*gen->mat[i][q]
                                         + mat->ten[i][j][l][q]*gen->mat[q][k]
                                         - mat->ten[q][i][k][l]*gen->mat[j][q]
                                         - mat->ten[i][j][k][q]*gen->mat[q][l]
                                         - gen->ten[q][j][k][l]*mat->mat[i][q]
                                         - gen->ten[i][j][l][q]*mat->mat[q][k]
                                         + gen->ten[q][i][k][l]*mat->mat[j][q]
                                         + gen->ten[i][j][k][q]*mat->mat[q][l];
                }
                // [eta(4),H(4)] (ommit O(6) terms)
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        kM->ten[i][j][k][l] += 2.0*(gen->ten[i][j][x][y]*mat->ten[y][x][k][l]
                                             - mat->ten[i][j][x][y]*gen->ten[y][x][k][l]);
                    }
                }

                // Enforce symmetry
                kM->ten[j][i][k][l] = -kM->ten[i][j][k][l];
                kM->ten[i][j][l][k] = -kM->ten[i][j][k][l];
                kM->ten[j][i][l][k] = kM->ten[i][j][k][l];
            }
        }
    }
}

void RKStep() {
    GENERATOR<<<nobOpt,tpbOpt>>>(master, gen);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[0], master);
    checkCudaSyncErr();

    DPSLOPE1<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[1], temp);
    checkCudaSyncErr();

    DPSLOPE2<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[2], temp);
    checkCudaSyncErr();

    DPSLOPE3<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[3], temp);
    checkCudaSyncErr();

    DPSLOPE4<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[4], temp);
    checkCudaSyncErr();

    SUMDP<<<nobOpt, tpbOpt>>>(kMat, master, ct);
    checkCudaSyncErr();
}

void embeddedRK() {
    double s = 0.0;
    copyToRecords(master, s, r);
    r++;
    checkHerm(master);
    #ifndef SUPPRESSOUTPUT
    printMatrix(master->mat, N);
    printH4interact(master);
    #endif

    while (s < l) {
        RKStep();
        s += h;
        #ifndef SUPPRESSOUTPUT
        printMatrix(master->mat, N);
        printH4interact(master);
        printf("\n");
        #endif
        copyToRecords(master, s, r);
        if (r < NUMRECORDS) r++;
        printf("s = %.4f, h = %.4f, r = %d\n", s, h, r);
    }
}

void DP () {
    GENERATOR<<<nobOpt,tpbOpt>>>(master, gen);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,prev);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[0], master);
    checkCudaSyncErr();

    DPSLOPE1<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[1], temp);
    checkCudaSyncErr();

    DPSLOPE2<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[2], temp);
    checkCudaSyncErr();

    DPSLOPE3<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[3], temp);
    checkCudaSyncErr();

    DPSLOPE4<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[4], temp);
    checkCudaSyncErr();

    DPSLOPE5<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[5], temp);
    checkCudaSyncErr();

    DPSLOPE6<<<nobOpt, tpbOpt>>>(kMat, temp, master);
    checkCudaSyncErr();

    CALCSLOPE<<<nobOpt,tpbOpt>>>(kMat[6], temp);
    checkCudaSyncErr();

    SUMDP<<<nobOpt, tpbOpt>>>(kMat, master, ct);
    checkCudaSyncErr();

    DPERROR<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
}

void embeddedDP () {
    double s = 0.0;
    double scale;
    double scaleprev = MAX_SCALE_FACTOR;
    double err, t;
    double qq;
    double tol;
    int last_interval = 0;
    int i, x, y, z, q;
    tol = etol/l;
    copyToRecords(master, s, r);
    r++;
    checkHerm(master);
    #ifndef SUPPRESSOUTPUT
    printMatrix(master->mat, N);
    printH4interact(master);
    #endif
    while (s < l) {
        scale = 1.0;
        // Will run a number of attempts just in case the h value is too large
        for (i = 0; i < ATTEMPTS; i++) {
            DP(); // Dormand-Prince method
            err = findMax(temp, &x, &y, &z, &q); // estimated error
            if (round(err) == err && round(err) == 0) {
                // zero error implies most accurate answer; will rarely happen
                scale = MAX_SCALE_FACTOR;
                printf("\n-- ERROR IS ZERO --\n");
                break;
            }
            // t is the point on the previous tensor which has the highest err
            t = readFloet(prev,x,y,z,q);
            // Choose qq = tol if t is zero. Else, qq = fabs(t)
            qq = (round(t) == t && round(t) == 0.0) ? tol : fabs(t);

            // Set the scale
            scale = 0.8 * sqrt( sqrt ( tol * qq / err ) );
            if (scale-scaleprev > 0.1) scale = scaleprev+0.1;
            scale = min( max(scale,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);

            // If the error is small enough, then no more attempts are required.
            if (err < (tol * qq)) break;

            // Update the scale accordingly
            h *= scale;
            if (s + h > l) h = l - s;
            else if (s + h + 0.5*h > l) h = 0.5 * h;

            // Restart the calculation.
            COPY<<<nob,tpb>>>(prev,master);
            checkCudaSyncErr();
        }
        if ( i >= ATTEMPTS ) { // Something here bothers me. Need to terminate until I fully understand this part.
            h = h * scale;
            tol = etol/(l-s);
            COPY<<<nob,tpb>>>(prev,master);
            checkCudaSyncErr();
            printf("-- REACHED MAX ATTEMPTS (h = %.5f, scale = %.3f)\n", h, scale);
            exit(0);
        } else {
            s += h;
            h *= scale;
            #ifndef SUPPRESSOUTPUT
            printMatrix(master->mat, N);
            printH4interact(master);
            printf("\n");
            #endif
            copyToRecords(master, s, r);
            if (r < NUMRECORDS) r++;
            if (last_interval) break;
            if (s + h > l) { last_interval = 1; h = l - s; }
            else if (s + h + 0.5*h > l) h = 0.5 * h;
            printf("%.4f,%.4f,%.4f,%d\n", s, h, scale, r);
            scaleprev = scale;
        }
    }
}

double runFES(struct Variables *v) {
    #ifndef SUPPRESSOUTPUT
    printf("Setting variables:\n");
    #endif
    setVariables(v);
    #ifndef SUPPRESSOUTPUT
    printf("Done\nInitializing:... ");
    #endif
    init();
    #ifndef SUPPRESSOUTPUT
    printf("Done\nOutput initial Hamiltonian:... ");
    #endif
    outputH2Qu("H2", master->mat, N);
    outputHamMathematica("i", master->mat,master->ten, N);
    #ifndef SUPPRESSOUTPUT
    printf("Done\nStarting simulation:\n");
    #endif
    startTime();
    embeddedDP();
    endTime();
    #ifndef SUPPRESSOUTPUT
    printf("Done\n");
    #endif
    outputHRecord("h", N, r, hRecord);
    outputDRecord("D", N, r, dRecord);
    outputDRecord("H2mat", N, r, H2Record);
    outputHamMathematica("f", master->mat,master->ten, N);
    return runTime();
}

double readFloet(struct floet *mat, int i, int j, int k, int l) {
    if (k == -1 && l == -1) return mat->mat[i][j];
    else                    return mat->ten[i][j][k][l];
}

////////////////////////////////////////////////////////////////////////////////
// Specific matrix based operations for the interacting model                 //
////////////////////////////////////////////////////////////////////////////////

void correctScale(double *scale, double prev, double dprev) {
    double c;
    if (*scale > prev) {
        if (prev-dprev > 0.0) c = min(*scale-prev,prev-dprev);
        else c = *scale-prev;
        if (c > 0.05) *scale = prev+0.01;
    }
}

int isDiag(struct floet *mat) {
    int c = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j && mat->mat[i][j] != 0.0) c = 0;
        }
    }
    return c;
}

double findMax(struct floet *mat, int *x, int *y, int *z, int *q) {
    double c = -1.0;
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

void copyToRecords(struct floet *mat, double t, int index) {
    if (index >= NUMRECORDS) return;
    for (int i = 0; i < N; i++) {
        hRecord[index]->h[i] = mat->mat[i][i];
        for (int j = 0; j < N; j++) {
            if (i == j) dRecord[index]->D[i][j] = mat->ten[i][j][i][j];
            else dRecord[index]->D[i][j] = -4.0*mat->ten[i][j][i][j];
            H2Record[index]->D[i][j] = mat->mat[i][j];
        }
    }
    hRecord[index]->t = t;
    dRecord[index]->t = t;
    H2Record[index]->t = t;
}

void printH4(struct floet *mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("H4[%d][%d]:\n",i,j);
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    printf("%.3f", mat->ten[i][j][k][l]);
                    if (l < N-1) printf(",");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

void printH4interact(struct floet *mat) {
    double num;
    for (int i = 0; i < N; i++) {
        if (i == 0) printf("D  =   ");
        else printf("       ");
        for (int j = 0; j < N; j++) {
            num = -4.0*mat->ten[i][j][i][j];
            if (i == j) printf("%.5f",mat->ten[i][j][i][j]);
            else if (num >= 0.0) printf("%.5f",num);
            else printf("%.4f",num);
            if (j != N-1) printf(", ");
        }
        printf("\n");
    }
    printf("-----------------------------------\n\n-----------------------------------");
}

void checkHerm(struct floet *mat) {
    if (!ISHERM(mat, N)) {
        printf("WARNING: Matrix not hermitian\n");
    }
}

void checkAHerm(struct floet *mat) {
    if (!ISANTIHERM(mat, N)) {
        printf("WARNING: Matrix not anti-hermitian\n");
    }
}

int TESTCONDITION(struct floet *mat) {
    int c = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (mat->mat[i][j] == 0.0) {
                printf("Condition error: (%d, %d) -> (%d, %d) ",i,j,j,i);
                if (i > j)  printf("(t) -> (b)\n");
                if (i < j)  printf("(b) -> (t)\n");
                if (i == j) printf("(d) -> (d)\n");
                c = 0;
            } else if (mat->mat[i][j] > 1.0) {
                printf("Condition error (SPOILED): (%d,%d)->(%d,%d)  ",i,j,j,i);
                if (i > j)  printf("(t) -> (b)\n");
                if (i < j)  printf("(b) -> (t)\n");
                if (i == j) printf("(d) -> (d)\n");
                c = 0;
            }
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (mat->ten[i][j][k][l] == 0.0 && !((i > j && k == l) || (i < j && k <= l))) {
                        printf("Condition error (IGNORED): (%d,%d,%d,%d)->(%d,%d,%d,%d)  ",i,j,k,l,l,k,j,i);
                        if (i > j) {
                            if (k > l)  printf("(tt) -> (bb)\n");
                            if (k < l)  printf("(tb) -> (tb)\n");
                            if (k == l) printf("(td) -> (db)\n");
                        } else if (i < j) {
                            if (k > l)  printf("(bt) -> (bt)\n");
                            if (k < l)  printf("(bb) -> (tt)\n");
                            if (k == l) printf("(bd) -> (dt)\n");
                        } else if (i == j) {
                            if (k > l)  printf("(dt) -> (bd)\n");
                            if (k < l)  printf("(db) -> (td)\n");
                            if (k == l) printf("(dd) -> (dd)\n");
                        }
                        c = 0;
                    }
                    if (mat->ten[i][j][k][l] > 1.0 && !((i > j && k == l) || (i < j && k <= l))) {
                        printf("Condition error (SPOILED): (%d,%d,%d,%d)->(%d,%d,%d,%d)  ",i,j,k,l,l,k,j,i);
                        if (i > j) {
                            if (k > l)  printf("(tt) -> (bb)\n");
                            if (k < l)  printf("(tb) -> (tb)\n");
                            if (k == l) printf("(td) -> (db)\n");
                        } else if (i < j) {
                            if (k > l)  printf("(bt) -> (bt)\n");
                            if (k < l)  printf("(bb) -> (tt)\n");
                            if (k == l) printf("(bd) -> (dt)\n");
                        } else if (i == j) {
                            if (k > l)  printf("(dt) -> (bd)\n");
                            if (k < l)  printf("(db) -> (td)\n");
                            if (k == l) printf("(dd) -> (dd)\n");
                        }
                        c = 0;
                    }
                }
            }
        }
    }
    printf("\n");
    return c;
}
