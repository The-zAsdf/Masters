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

#define NUMRECORDS 200
#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

// #define DEBUG

__managed__ floet *master;
__managed__ floet *prev;
__managed__ floet *temp;
__managed__ floet *gen;
__managed__ floet **kMat;
__managed__ float **invGausJ;
__managed__ float **invGausD;
__managed__ float *uniform;
__managed__ ind **threadIndex;

__managed__ float W;
__managed__ float J;
__managed__ int N;
__managed__ int numElem;    // Number of elements
__managed__ float h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
double l;                   // Total simulation steps

floardH **hRecord;
floardD **dRecord;
floardF **iRecord;
int r;

__global__ void generateMaster(curandState_t* states) {
    int i, j, k, l, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (k != -1 && (i == j && k == l) && i > k) { // init :D:_{ij}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            // NOT CORRECT. DOUBLE CHECK //
            master->ten[i][i][k][k] = invGausD[abs(i-k)][r]/2;
            master->ten[k][k][i][i] = invGausD[abs(i-k)][r]/2;
            master->ten[i][k][k][i] = -invGausD[abs(i-k)][r]/2;
            master->ten[k][i][i][k] = -invGausD[abs(i-k)][r]/2;
        } else if (k != -1 && l != -1 && (i != l && j != k)) { // init :G:_{ijkl}
            // PLACEHOLDER //
            // r = (int)(curand_uniform(&states[id])*((float) 1000));
            // master->ten[i][j][k][l] = invGausJ[abs(i-j)][r];
            master->ten[i][j][k][l] = 0.0f;
        } else if (k == -1 && (i == j)) { // init h_i
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            master->mat[i][j] = (uniform[r]-0.5f)*2.0f*W;
        } else if (k == -1 && (i > j)){ // init :J:_{ij}
            // PLACEHOLDER //
            r = (int)(curand_uniform(&states[id])*1001 - 1);
            master->mat[i][j] = invGausJ[abs(i-j)][r];
            master->mat[j][i] = invGausJ[abs(i-j)][r];
        }
    }


}

void setVariables(struct Variables *v) {
    // cudaDeviceProp props;
    // int deviceId;
    W = v->W;
    J = v->J;
    N = v->N[v->index];
    h = v->h;
    l = v->steps;

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
    err = cudaMallocManaged(&uniform, sizeof(float)*1000);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGausJ, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&invGausD, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&invGausJ[i], sizeof(float)*1000);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&invGausD[i], sizeof(float)*1000);
        if (err != cudaSuccess) CUDAERROR(err);
    }
    for (int i = 0; i < 1000; i++) uniform[i] = ((float) i)/((float) (1000-1));
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
            invGausJ[i][j] = gaussianICDF(uniform[r], (float) i+1, J, 1.0f);
            r = rand()%1000;
            invGausD[i][j] = gaussianICDF(uniform[r], (float) i+1, J*0.1f, 1.0f);
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
        hRecord[i]->h = (float*) malloc(sizeof(float)*N);
        dRecord[i] = (struct floardD *)malloc(sizeof(struct floardD));
        dRecord[i]->D = (float**) malloc(sizeof(float*)*N);
        for (int j = 0; j < N; j++) {
            dRecord[i]->D[j] = (float*)malloc(sizeof(float)*N);
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

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (k == -1 && l == -1) {
            kM->mat[i][j] = 0.0f;
            for (int q = 0; q < N; q++) {
                kM->mat[i][j] += gen->mat[i][q]*mat->mat[q][j]
                               - mat->mat[i][q]*gen->mat[q][j];
            }
        } else {
            kM->ten[i][j][k][l] = 0.0f;
            for (int q = 0; q < N; q++) {
                kM->ten[i][j][k][l] += gen->ten[i][q][k][l]*mat->mat[q][j]
                                     + gen->ten[i][j][k][q]*mat->mat[q][l]
                                     - gen->ten[i][j][q][l]*mat->mat[k][q]
                                     - gen->ten[q][j][k][l]*mat->mat[i][q]
                                     + mat->ten[q][j][k][l]*gen->mat[i][q]
                                     + mat->ten[i][j][q][l]*gen->mat[k][q]
                                     - mat->ten[i][j][k][q]*gen->mat[q][l]
                                     - mat->ten[i][q][k][l]*gen->mat[q][j];

            }
        }
    }
}

void DP () {
    GENERATOR<<<nob, tpb>>>(master, gen);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();


    COPY<<<nob,tpb>>>(master,prev);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[0], temp);
    checkCudaSyncErr();

    DPSLOPE1<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[1], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    DPSLOPE2<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[2], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    DPSLOPE3<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[3], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    DPSLOPE4<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[4], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    DPSLOPE5<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[5], temp);
    checkCudaSyncErr();

    COPY<<<nob,tpb>>>(master,temp);
    checkCudaSyncErr();

    DPSLOPE6<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();

    CALCSLOPE<<<nob,tpb>>>(kMat[6], temp);
    checkCudaSyncErr();

    SUMDP<<<nob,tpb>>>(kMat, master);
    checkCudaSyncErr();

    DPERROR<<<nob, tpb>>>(kMat, temp);
    checkCudaSyncErr();
}

void embeddedDP () {
    double s = 0.0;
    double scale;
    float err, t;
    double qq;
    double tol, etol;
    int last_interval = 0;
    int i, x, y, z, q;
    etol = 0.001;
    tol = etol/l;
    copyToRecords(master, s, r);
    printMatrix(master->mat, N);
    printH4interact(master);
    while (s < l) {
        scale = 1.0;
        for (i = 0; i < ATTEMPTS; i++) {
            #ifdef DEBUG
                printf("(%d) Starting DP: ", i);
            #endif
            DP();
            #ifdef DEBUG
                printf("DP done");
            #endif
            err = findMax(temp, &x, &y, &z, &q);
            if (roundf(err) == err && roundf(err) == 0) {
                scale = MAX_SCALE_FACTOR;
                printf("\n-- ERROR IS ZERO --\n");
                printMatrix(temp->mat, N);
                printH4interact(temp);
                printf("-------------------");
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
            #ifdef DEBUG
                printf("(h = %.5f)\n", h);
            #endif
            COPY<<<nob,tpb>>>(prev,master);
            checkCudaSyncErr();

        }
        if ( i >= ATTEMPTS ) {
            h = h * scale;
            tol = etol/(l-s);
            COPY<<<nob,tpb>>>(prev,master);
            checkCudaSyncErr();
            printf("-- REACHED MAX ATTEMPTS (h = %.5f, scale = %.3f)\n", h, scale);
        } else {
            printf("s = %.4f, h = %.4f, scale = %.4f\n", s, h, scale);
            s += h;
            h *= scale;
            if ( last_interval ) break;
            if (s + (double) h > l) { last_interval = 1; h = (float) l - (float) s; }
            else if (s + h + 0.5*h > l) h = 0.5 * h;
            printMatrix(master->mat, N);
            printH4interact(master);
            printf("\n");
            copyToRecords(master, s, r);
            if (r < NUMRECORDS) r++;
        }
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
    outputHRecord("h", N, r, hRecord);
    outputDRecord("D", N, r, dRecord);
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

float calcInvariant() {
    float t = 0.0f;
    for (int i = 0; i < N; i++) {
        t += powf(master->mat[i][i], 2.0f);

        for (int j = 0; j < N; j++) {
            t += 0.5f*powf(master->mat[i][j],2.0f) + master->ten[i][i][j][j]
                                                   - master->ten[i][j][j][i];
        }
    }
    return t;
}

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

void copyToRecords(struct floet *mat, double t, int index) {
    if (index >= NUMRECORDS) return;
    for (int i = 0; i < N; i++) {
        hRecord[index]->h[i] = mat->mat[i][i];
        for (int j = 0; j < N; j++) {
            dRecord[index]->D[i][j] = mat->ten[i][i][j][j] - mat->ten[i][j][j][i];
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
    for (int j = 0; j < N; j++) {
        printf("D[0][%d]: %.3f ,\n",j,mat->ten[0][0][j][j] - mat->ten[0][j][j][0]);
    }
    printf("\n");
}
