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

__managed__ floet **master;
__managed__ floet **prev;
__managed__ floet **temp;
__managed__ floet ***kMat;
__managed__ float **invGaus;
__managed__ float *uniform;
__managed__ ind **threadIndex;
__managed__ ind **threadIndexJJ;

__managed__ float W;
__managed__ float J;
__managed__ int N;
__managed__ int triangN;
__managed__ int numElem;    // Number of elements
__managed__ float h;
__managed__ size_t tpb;     // Threads per block
__managed__ size_t nob;     // Number of blocks
__managed__ size_t tpbHH;   // Threads per block
__managed__ size_t nobHH;   // Number of blocks
__managed__ size_t tpbJJ;   // Threads per block
__managed__ size_t nobJJ;   // Number of blocks
double l;                  // Total simulation steps



__global__ void generateMaster(curandState_t* states) {
    int i, j, k, l, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;
        if (l == -1) { // init :D:_{ij}
            r = (int)(curand_uniform(&states[id])*((float) numElem));
            master[i]->d[j] = invGaus[abs(i-j)][r];
        } else if (l == -2) { // init :J:_{ij}
            r = (int)(curand_uniform(&states[id])*((float) numElem));
            master[i]->j[j] = invGaus[abs(i-j)][r];
        } else if (l == -3) { // init h_i
            master[i]->h = curand_uniform(&states[id])*W;
        } else { // init :G:_{ijkl}
            master[i]->g[j][k][l] = 0.0f;
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
        numElem++; // each h;
        for (int j = 0; j < N; j++) numElem++; // each :D:
        for (int j = 0; j < N-i; j++) {
            numElem++; // each :J:

            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) numElem++; // each :G:
            }
        }
    }
    triangN = N(N+1)/2;
    determineThreadsAndBlocks();
    determineThreadsAndBlocksHH();
    determineThreadsAndBlocksJJ();
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

size_t calculateBlocksHH(size_t threads) {
    size_t blocks = 1;
    for (size_t i = threads; i < N; i += threads) {
        if (blocks < threads*2 || threads == 1024) {
            blocks++;
        } else {
            return 0;
        }
    }
    return blocks;
}

size_t calculateBlocksJJ(size_t threads) {
    size_t blocks = 1;
    for (size_t i = threads; i < N*(N+1)/2; i += threads) {
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

void determineThreadsAndBlocksHH() {
    cudaDeviceProp props;
    int deviceId;
    size_t blocks;
    size_t threads = 0;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    do {
        threads += props.warpSize;
        blocks = calculateBlocksHH(threads);
    } while (blocks == 0 && threads < 1024);
    nobHH = blocks;
    tpbHH = threads;
}

void determineThreadsAndBlocksJJ() {
    cudaDeviceProp props;
    int deviceId;
    size_t blocks;
    size_t threads = 0;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    do {
        threads += props.warpSize;
        blocks = calculateBlocksJJ(threads);
    } while (blocks == 0 && threads < 1024);
    nobJJ = blocks;
    tpbJJ = threads;
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
    #ifdef MEMDEBUG
    printf("\n\tAllocating master: ");
    #endif
    err = cudaMallocManaged(&master, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&master[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&master[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&master[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tAllocating prev: ");
    #endif

    err = cudaMallocManaged(&prev, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&prev[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&prev[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&prev[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tAllocating temp: ");
    #endif
    err = cudaMallocManaged(&temp, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&temp[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&temp[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&temp[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tAllocating kMat: ");
    #endif
    err = cudaMallocManaged(&kMat, sizeof(struct floet***)*7);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < 7; i++) {
        err = cudaMallocManaged(&kMat[i], sizeof(struct floet**)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i][j], sizeof(struct floet*));
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i][j]->d, sizeof(float)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i][j]->j, sizeof(float)*(N-j));
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i][j]->g, sizeof(float *)*(N-j));
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N-j; k++) {
                err = cudaMallocManaged(&kMat[i][j]->g[k], sizeof(float *)*N);
                if (err != cudaSuccess) CUDAERROR(err);
                for (int l = 0; l < N; l++) {
                    err = cudaMallocManaged(&kMat[i][j]->g[k][l], sizeof(float)*N);
                    if (err != cudaSuccess) CUDAERROR(err);
                }
            }
        }
    }
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tAllocating threadIndex: ");
    #endif
    err = cudaMallocManaged(&threadIndex, sizeof(struct index*)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&threadIndex[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }

    err = cudaMallocManaged(&threadIndexJJ, sizeof(struct index*)*N*(N+1)/2);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < N*(N+1)/2; i++) {
        err = cudaMallocManaged(&threadIndexJJ[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tInitializing threadIndex: ");
    #endif

    // threadIndex. Each thread corresponds to a matrix element.
    count = 0;
    for (int i = 0; i < N; i++) {
        // each h element has a thread
        threadIndex[count]->i = i;
        threadIndex[count]->j = -1;
        threadIndex[count]->k = -1;
        threadIndex[count]->l = -3;
        count++;
        for (int j = 0; j < N; j++) {
            // each :D: element has a thread
            threadIndex[count]->i = i;
            threadIndex[count]->j = j;
            threadIndex[count]->k = -1;
            threadIndex[count]->l = -1;
            count++;
        }
        for (int j = 0; j < N-i; j++) {
            // each J element has a thread
            threadIndex[count]->i = i;
            threadIndex[count]->j = j;
            threadIndex[count]->k = -1;
            threadIndex[count]->l = -2;
            count++;

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

    count = 0;
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N-i; j++) {
            threadIndexJJ[count]->i = i;
            threadIndexJJ[count]->j = j;
            threadIndexJJ[count]->k = -1;
            threadIndexJJ[count]->l = -1;
            count++;
        }
    }
    #ifdef MEMDEBUG
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
    #ifdef MEMDEBUG
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
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tgenerateMaster: ");
    #endif

    // initialize master values
    generateMaster<<<nob,tpb>>>(states);
    checkCudaSyncErr();
    #ifdef MEMDEBUG
    printf("Done\n");

    printf("\tfreeing states and distributions: ");
    #endif

    // free distribution + cuRAND states
    cudaFree(states);
    cudaFree(invGaus);
    cudaFree(uniform);
    #ifdef MEMDEBUG
    printf("Done\n");
    #endif

}

void freeMem() {
    for (int i = 0; i < N; i++) {
        cudaFree(master[i]->d);
        cudaFree(master[i]->j);

        cudaFree(prev[i]->d);
        cudaFree(prev[i]->j);

        cudaFree(temp[i]->d);
        cudaFree(temp[i]->j);
        for (int j = 0; j < N-i; j++) {
            for (int k = 0; k < N; k++) {
                cudaFree(master[i]->g[j][k]);
                cudaFree(prev[i]->g[j][k]);
                cudaFree(temp[i]->g[j][k]);
            }
            cudaFree(master[i]->g[j]);
            cudaFree(prev[i]->g[j]);
            cudaFree(temp[i]->g[j]);
        }
        cudaFree(master[i]->g);
        cudaFree(prev[i]->g);
        cudaFree(temp[i]->g);
    }
    cudaFree(master);
    cudaFree(temp);
    cudaFree(prev);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < N; j++) {
            cudaFree(kMat[i][j]->d);
            cudaFree(kMat[i][j]->j);

            for (int k = 0; k < N-j; k++) {
                for (int l = 0; l < N; l++) {
                    cudaFree(kMat[i][j]->g[k][l]);
                }
                cudaFree(kMat[i][j]->g[k]);
            }
            cudaFree(kMat[i][j]->g);
        }
        cudaFree(kMat[i]);
    }
    cudaFree(kMat);
}

// Keep these functions in the same file as CALCSLOPE, or find a way to
// (efficiently) pass device code through kernal (i.e no memcopies)
__device__ void funcH(struct floet **mat, float *q, int i) {

}

__device__ void funcJ(struct floet **mat, float *q, int i, int j) {

}

__device__ void funcD(struct floet **mat, float *q, int i, int j) {

}

__device__ void funcG(struct floet **mat, float *q, int i, int j, int k, int l) {

}

// Optimize this for MASSIVE performance increase (optimized matrix multiplication)
__global__ void CALCSLOPE(floet **kM, floet **mat) {
    int i, j, k, l;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        k = threadIndex[id]->k;
        l = threadIndex[id]->l;

        if (l == -1) { // funcD
            funcD(mat, &kM[i]->d[j], i, j);
        } else if (l == -2) { // funcJ
            funcJ(mat, &kM[i]->j[j], i, j);
        } else if (l == -3) { // funcH
            funcH(mat, &kM[i]->h, i);
        } else { // funcG
            funcG(mat, &kM[i]->g[j][k][l], i, j, k, l);
        }
    }
}

void DP () {
    // Copy master into temp
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
    double tol = 0.001/l;
    int last_interval = 0;
    int i, x, y, z, q;
    while (s < l) {
        scale = 1.0;
        for (i = 0; i < ATTEMPTS; i++) {
            DP();
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

float readFloet(struct floet **mat, int i, int j, int k, int l) {
    if (l == -1) {
        return mat[i]->d[j];
    } else if (l == -2) {
        return mat[i]->j[j];
    } else if (l == -3) {
        return mat[i]->h;
    } else {
        return mat[i]->g[j][k][l];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Specific matrix based operations for the interacting model                 //
////////////////////////////////////////////////////////////////////////////////

float findMax(floet **mat, int *x, int *y, int *z, int *q) {
    float c = -1.0f;
    for (int i = 0; i < N; i++) {
        if (mat[i]-> h > c) {
            c = mat[i]->h;
            *x = i;
            *y = -1;
            *z = -1;
            *q = -3;
        }
        for (int j = 0; j < N; j++) {
            if (mat[i]-> d[j] > c) {
                c = mat[i]->d[j];
                *x = i;
                *y = j;
                *z = -1;
                *q = -1;
            }
        }
        for (int j = 0; j < N-i; j++) {
            if (mat[i]-> j[j] > c) {
                c = mat[i]->j[j];
                *x = i;
                *y = j;
                *z = -1;
                *q = -2;
            }
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    if (mat[i]->g[j][k][l] > c) {
                        c = mat[i]->g[j][k][l];
                        *x = i;
                        *y = j;
                        *z = k;
                        *q = l;
                    }
                }
            }
        }
    }
    return c;
}


float flowInvariant(floet **mat) {
    float t = 0.0f;
    for (int i = 0; i < N; i++) {
        t += powf(mat[i]->h,2.0f);

        for (int j = 0; j < N-i; j++) {
            t += 0.5f*powf(mat[i]->j[j], 2.0f);

            // Unsure how to calculate :G:_{ijkl}
            // for (int k = 0; k < N; k++) {
            //     for (int l = 0; l <  N; l++) {
            //         t += powf(mat[i]->g[j][k][l], 2.0f);
            //     }
            // }
        }

        for (int j = 0; j < N; j++) {
            t += powf(mat[i]->d[j], 2.0f);
        }
    }
    return temp;
}
