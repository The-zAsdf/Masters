#include <stdio.h>

#define CUDAERROR(err) printCudaErr(err)

typedef struct index{
    int i;
    int j;
    int x;
    int y;
} ind;

typedef struct floet {
    float h;
    float *d;
    float *j;
    float ***g;
}f;

void printCudaErr(cudaError_t err) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    exit(1);
}
