#include <stdlib.h>
#include <stdio.h>
#include "err.h"

void checkCudaSyncErr() {
    cudaError_t syncErr = cudaGetLastError();
    cudaError_t asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) CUDAERROR(syncErr);
    if (asyncErr != cudaSuccess) CUDAERROR(asyncErr);
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
