#include <stdlib.h>
#include <stdio.h>
#include "err.cuh"

void checkCudaSyncErr() {
    cudaError_t syncErr = cudaGetLastError();
    cudaError_t asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printCudaSyncErr(syncErr);
    if (asyncErr != cudaSuccess) printCudaAsyncErr(asyncErr);
}

void printErr(int flag) {
    switch (flag) {
        case 1:
            fprintf(stderr, "Error: could not allocate memory\n");
            exit(1);
        case 2:
            fprintf(stderr, "Error: No filename found\n");
            exit(2);
    }
}

void printCudaErr(cudaError_t err) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    exit(1);
}

void printCudaAsyncErr(cudaError_t err) {
    fprintf(stderr, "Error (Async): %s\n", cudaGetErrorString(err));
    exit(1);
}

void printCudaSyncErr(cudaError_t err) {
    fprintf(stderr, "Error (Sync): %s\n", cudaGetErrorString(err));
    exit(1);
}
