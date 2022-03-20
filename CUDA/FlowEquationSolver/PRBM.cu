#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "measureTime.h"
#include "IO.h"
#include "err.cuh"
#include "PRBM.h"
#include "distribution.h"
#include "matOperations.cuh"

#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

__managed__ float **master;
__managed__ float **prev;
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
double l;                  // Total simulation steps

float findMax(float **mat, int *x, int *y) {
    float c = mat[0][0];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            if (c < mat[i][j] && (i != 0 || j != 0)) {
                c = mat[i][j];
                *x = i;
                *y = j;
            }
        }
    }
    return c;
}

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
    l = v->steps;

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

    err = cudaMallocManaged(&prev, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&temp, sizeof(float*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    err = cudaMallocManaged(&kMat, sizeof(float*)*7);
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

        err = cudaMallocManaged(&prev[i], sizeof(float)*(N-i));
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

    // kMat
    for (int i = 0; i < 7; i++) {
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
        cudaFree(prev[i]);
    }
    cudaFree(master);
    cudaFree(temp);
    cudaFree(prev);

    for (int i = 0; i < 7; i++) {
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
    float hi = mat[i][0];
    *q = 0.0f;
    for (int k = 0; k < N; k++) {
        if (i != k) {
            *q += powf(mat[min(i,k)][abs(i-k)], 2.0f)*(hi-mat[k][0]);
        }
    }
    *q *= 2.0f;
}

__device__ void funcJ(float **mat, float *q, int i, int j) {
    float hi, hj;
    int x = i;
    int y = j+i;
    *q = 0.0f;
    hi = mat[x][0];
    hj = mat[y][0];
    for (int k = 0; k < N; k++) {
        if (x != k && y != k) {
            *q -= mat[min(x,k)][abs(x-k)]*mat[min(y,k)][abs(y-k)]*(2.0f*mat[k][0]-hi-hj);
        }
    }
    if (x != y) *q -= mat[i][j]*powf(hi-hj,2.0f);
}

__global__ void CALCSLOPE(float **kM, float **mat) {
    int i, j;
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

void RK4() {
    double s = 0.0;
    while (s < l) {
        // Copy master into temp
        COPY<<<nob,tpb>>>(master,temp);
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

        COPY<<<nob,tpb>>>(master,temp);
        checkCudaSyncErr();

        SUMRK<<<nob,tpb>>>(kMat, master);
        checkCudaSyncErr();
        // printf("Final master:\n");
        printf("s = %.4f\n", s);
        printMatrix(master, N);
        printf("\n");
        s += (double) h;
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
    float err;
    double qq;
    double tol = 0.001/l;
    int last_interval = 0;
    int i, x, y;
    while (s < l) {
        scale = 1.0;
        for (i = 0; i < ATTEMPTS; i++) {
            DP();
            err = findMax(temp, &x, &y);
            if (roundf(err) == err && roundf(err) == 0) {
                scale = MAX_SCALE_FACTOR;
                break;
            }
            if (roundf(prev[x][y]) == prev[x][y] && roundf(prev[x][y]) == 0.0f) {
                qq = tol;
            } else {
                qq = fabsf(prev[x][y]);
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
        printMatrix(master, N);
        printf("\n");
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
