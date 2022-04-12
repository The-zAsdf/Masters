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

__managed__ floet *master;
__managed__ floet *prev;
__managed__ floet *temp;
__managed__ floet **kMat;
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
    int i, j, x, y, r;
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if (id < numElem) {
        i = threadIndex[id]->i;
        j = threadIndex[id]->j;
        x = threadIndex[id]->x;
        y = threadIndex[id]->y;
        if (j == 0) {
            if (x == -1 && y == -1) { // init h_i
                master[i][0]->el = curand_uniform(&states[id])*W;
            } else if (y == -1) { // init :D:_{ix}

            } else { // error

            }
        } else {
            if (x == -1 && y == -1) { // init J_{ij}
                r = (int)(curand_uniform(&states[id])*((float) numElem));
                master[i][j]->el = invGaus[abs(i-j)][r];
            } else { // init :G:_{ijkl}

            }
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
            if (j == 0) { // Diag elements
                for (int x = 0; x < N; x++) {
                    numElem++;
                }
            } else { // Off diag elements
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        numElem++;
                    }
                }
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
    err = cudaMallocManaged(&master, sizeof(struct floet*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&master[i]->d, sizeof(struct diag));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&master[i]->o, sizeof(struct odiag)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        master[i]->n = N-i;
    }

    // initialize float lists inside each element
    for (int i = 0; i < N; i++) {
        // Diagonal elements
        err = cudaMallocManaged(&master[i]->d->mel, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        // Off diagonal elements
        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&master[i]->o[j]->mel, sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&master[i]->o[j]->mel[k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }

    // Allocating prev. Same as above
    err = cudaMallocManaged(&prev, sizeof(struct floet*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&prev[i]->d, sizeof(struct diag));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&prev[i]->o, sizeof(struct odiag)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        prev[i]->n = N-i;
    }

    // initialize float lists inside each element
    for (int i = 0; i < N; i++) {
        // Diagonal elements
        err = cudaMallocManaged(&prev[i]->d->mel, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        // Off diagonal elements
        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&prev[i]->o[j]->mel, sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&prev[i]->o[j]->mel[k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }

    // Allocating temp. Same as above
    err = cudaMallocManaged(&temp, sizeof(struct floet*)*N);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&temp[i]->d, sizeof(struct diag));
        if (err != cudaSuccess) CUDAERROR(err);

        err = cudaMallocManaged(&temp[i]->o, sizeof(struct odiag)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);

        temp[i]->n = N-i;
    }

    // initialize float lists inside each element
    for (int i = 0; i < N; i++) {
        // Diagonal elements
        err = cudaMallocManaged(&temp[i]->d->mel, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        // Off diagonal elements
        for (int j = 0; j < N-i; j++) {
            err = cudaMallocManaged(&temp[i]->o[j]->mel, sizeof(float*)*N);
            if (err != cudaSuccess) CUDAERROR(err);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&temp[i]->o[j]->mel[k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
            }
        }
    }

    // Allocating kMat. Same as above
    err = cudaMallocManaged(&kMat, sizeof(struct floet*)*7);
    if (err != cudaSuccess) CUDAERROR(err);

    for (int i = 0; i < 7; i++) {
        err = cudaMallocManaged(&kMat[i], sizeof(struct floet*)*N);
        if (err != cudaSuccess) CUDAERROR(err);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i][j]->d, sizeof(struct diag));
            if (err != cudaSuccess) CUDAERROR(err);

            err = cudaMallocManaged(&kMat[i][j]->o, sizeof(struct odiag)*(N-j));
            if (err != cudaSuccess) CUDAERROR(err);

            kMat[i][j]->n = N-j;
        }
    }

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N-j; k++) {
                if (k == 0) {
                    err = cudaMallocManaged(&kMat[i][j][k]->mel, sizeof(float)*N);
                    if (err != cudaSuccess) CUDAERROR(err);
                } else {
                    err = cudaMallocManaged(&kMat[i][j][k]->mel, sizeof(float)*N);
                    if (err != cudaSuccess) CUDAERROR(err);

                    for (int l = 0; l < N; l++) {
                        err = cudaMallocManaged(&kMat[i][j][k]->mel[l], sizeof(float)*N);
                        if (err != cudaSuccess) CUDAERROR(err);
                    }
                }
            }
        }
    }

    err = cudaMallocManaged(&threadIndex, sizeof(struct index*)*numElem);
    if (err != cudaSuccess) CUDAERROR(err);
    for (int i = 0; i < numElem; i++) {
        err = cudaMallocManaged(&threadIndex[i], sizeof(struct index));
        if (err != cudaSuccess) CUDAERROR(err);
    }

    // threadIndex. Each thread corresponds to a matrix element.
    count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            threadIndex[count]->i = i;
            threadIndex[count]->j = j;
            threadIndex[count]->x = -1;
            threadIndex[count]->y = -1;
            count++;
            if (j == 0) { // Diag elements
                for (int x = 0; x < N; x++) {
                    threadIndex[count]->i = i;
                    threadIndex[count]->j = 0;
                    threadIndex[count]->x = x;
                    threadIndex[count]->y = -1;
                    count++;
                }
            } else { // Off diag elements
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        threadIndex[count]->i = i;
                        threadIndex[count]->j = j;
                        threadIndex[count]->x = x;
                        threadIndex[count]->y = y;
                        count++;
                    }
                }
            }
        }
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

}

__device__ void funcJ(float **mat, float *q, int i, int j) {

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
