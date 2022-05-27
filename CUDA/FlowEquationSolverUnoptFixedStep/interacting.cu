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

#define NUMRECORDS 400
#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0
#define _USE_MATH_DEFINES
#include <math.h>

// #define DEBUG
// #define TESTDEBUG
// #define OUTPUTMAT
// #define DPDEBUG



__managed__ floet *master;
__managed__ floet *prev;
__managed__ floet *temp;
__managed__ floet *gen;
__managed__ floet **kMat;
__managed__ double **invGausJ;
__managed__ double **invGausD;
__managed__ double *uniform;
__managed__ ind **threadIndex;

__managed__ double W;
__managed__ double D;
__managed__ double etol;
__managed__ double J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ double h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
double l;                   // Total simulation steps

floardH **hRecord;
floardD **dRecord;
floardF **iRecord;
int r;
int count;

// const char * const H2FileName[] = {"H2_0","H2_1","H2_2","H2_3","H2_4","H2_5","H2_6","H2_7","H2_8","H2_9","H2_10"};
// const char * const H4FileName[] = {"H4_0","H4_1","H4_2","H4_3","H4_4","H4_5","H4_6","H4_7","H4_8","H4_9","H4_10"};
// const char * const dH2FileName[] = {"dH2_0","dH2_1","dH2_2","dH2_3","dH2_4","dH2_5","dH2_6","dH2_7","dH2_8","dH2_9","dH2_10"};
// const char * const dH4FileName[] = {"dH4_0","dH4_1","dH4_2","dH4_3","dH4_4","dH4_5","dH4_6","dH4_7","dH4_8","dH4_9","dH4_10"};
// const char * const eta2FileName[] = {"eta2_0","eta2_1","eta2_2","eta2_3","eta2_4","eta2_5","eta2_6","eta2_7","eta2_8","eta2_9","eta2_10"};
// const char * const eta4FileName[] = {"eta4_0","eta4_1","eta4_2","eta4_3","eta4_4","eta4_5","eta4_6","eta4_7","eta4_8","eta4_9","eta4_10"};

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

__global__ void generateMasterH4(curandState_t* states) {
    int i, j, k, l, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    // double phi;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k != -1 && l != -1) { // init :D:_{ij}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            if ( i == k + 1 || k == i + 1 || i == l + 1 || l == i + 1) {
                if (i == k && j == l) {
                    master->ten[i][j][k][l] = -D;
                }
                if (i == l && j == k) {
                    master->ten[i][j][k][l] = D;
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
    RESET<<<nob,tpb>>>(master);
    checkCudaSyncErr();
    generateMasterH4<<<nob,tpb>>>(states);
    checkCudaSyncErr();
    inputMaster();
    #ifdef DEBUG
    printf("Done\n");

    printf("\tfreeing states and distributions: ");
    #endif

    // free distribution + cuRAND states
    cudaFree(states);
    cudaFree(invGausJ);
    cudaFree(invGausD);
    cudaFree(uniform);
    #ifdef DEBUG
    printf("Done\n");

    printf("\tAllocating record objects:");
    #endif

    hRecord = (struct floardH**)malloc(sizeof(struct floardH*)*NUMRECORDS);
    dRecord = (struct floardD**)malloc(sizeof(struct floardD*)*NUMRECORDS);
    iRecord = (struct floardF**)malloc(sizeof(struct floardF*)*NUMRECORDS);

    for (int i = 0; i < NUMRECORDS; i++) {
        hRecord[i] = (struct floardH *)malloc(sizeof(struct floardH));
        iRecord[i] = (struct floardF *)malloc(sizeof(struct floardF));
        hRecord[i]->h = (double*) malloc(sizeof(double)*N);
        dRecord[i] = (struct floardD *)malloc(sizeof(struct floardD));
        dRecord[i]->D = (double**) malloc(sizeof(double*)*N);
        for (int j = 0; j < N; j++) {
            dRecord[i]->D[j] = (double*)malloc(sizeof(double)*N);
        }
    }

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

    free(hRecord);
    free(dRecord);
}


// Optimize this for MASSIVE performance increase (optimized matrix multiplication)
__global__ void CALCSLOPE(struct floet *kM, struct floet *mat) {
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    double num;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1 && i >= j) {
            kM->mat[i][j] = 0.0;
            // [eta(2),H(2)]
            for (int q = 0; q < N; q++) {
                kM->mat[i][j] += gen->mat[i][q]*mat->mat[q][j]
                               - mat->mat[i][q]*gen->mat[q][j];
            }
            if (i != j) kM->mat[j][i] = kM->mat[i][j];
        } else if (k != -1 && l != -1) {
            kM->ten[i][j][k][l] = 0.0;
            if (i > j && k > l) {
                num = 0.0;
                // [eta(2),H(4)] + [eta(4),H(2)]
                for (int q = 0; q < N; q++) {
                     num += mat->ten[q][j][k][l]*gen->mat[i][q]
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
                        num += 2.0*(gen->ten[i][j][x][y]*mat->ten[y][x][k][l]
                                             - mat->ten[i][j][x][y]*gen->ten[y][x][k][l]);
                    }
                }

                // Enforce symmetry
                kM->ten[i][j][k][l] = num;
                kM->ten[j][i][k][l] = -num;
                kM->ten[i][j][l][k] = -num;
                kM->ten[j][i][l][k] = num;
            }
        }
    }
}

// void NEXT(int step) {
void NEXT() {
    RESET<<<nob, tpb>>>(gen);
    checkCudaSyncErr();
    // outputH2(H2FileName[step], master->mat,N);
    // outputH4(H4FileName[step], master->ten,N);

    GENERATOR<<<nob, tpb>>>(master, gen);
    checkCudaSyncErr();
    // outputH2(eta2FileName[step], gen->mat,N);
    // outputH4(eta4FileName[step], gen->ten,N);

    CALCSLOPE<<<nob,tpb>>>(temp, master);
    checkCudaSyncErr();
    // outputH2(dH2FileName[step], temp->mat,N);
    // outputH4(dH4FileName[step], temp->ten,N);

    ADDSLOPE<<<nob,tpb>>>(master, temp, h);
    checkCudaSyncErr();
}

void embedded () {
    double s = 0.0;
    checkHerm(master);
    printMatrix(master->mat, N);
    printH4(master);
    printH4interact(master);
    copyToRecords(master,s,r);
    for (int i = 0; i < NUMRECORDS && s < l; i++) {
        NEXT();
        s += h;
        if (r < NUMRECORDS) r++;
        checkHerm(master);
        printf("s = %.4f\n", s);
        printMatrix(master->mat, N);
        printH4interact(master);
        copyToRecords(master,s,r);
    }
}

double runPRBM(struct Variables *v) {
    printf("Setting variables:\n");
    setVariables(v);
    printf("Done\nInitializing:... ");
    init();
    printf("Done\nStarting simulation:\n");
    startTime();
    embedded();

    endTime();
    outputHRecord("h", N, r, hRecord);
    outputDRecord("D", N, r, dRecord);
    outputiRecord("i", N, r, iRecord);
    printf("Done\n");
    freeMem();
    return runTime();
}

double readFloet(struct floet *mat, int i, int j, int k, int l) {
    if (k == -1 && l == -1) return mat->mat[i][j];
    else                    return mat->ten[i][j][k][l];
}

////////////////////////////////////////////////////////////////////////////////
// Specific matrix based operations for the interacting model                 //
////////////////////////////////////////////////////////////////////////////////

double calcInvariant() {
    double t = 0.0;
    for (int i = 0; i < N; i++) {
        t += pow(master->mat[i][i], 2.0);

        for (int j = 0; j < N; j++) {
            if (i != j) {
                t += 0.5*pow(master->mat[i][j],2.0) + 4.0*pow(master->ten[i][j][i][j],2.0);
            }
        }
    }
    return t;
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
            dRecord[index]->D[i][j] = mat->ten[i][j][i][j];
        }
    }
    iRecord[index]->f = calcInvariant();
    hRecord[index]->t = t;
    dRecord[index]->t = t;
    iRecord[index]->t = t;
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
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) printf("D[%d][%d]: %.5f ,\n",i,j,mat->ten[i][j][i][j]);
        }
    }
    printf("\n");

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
