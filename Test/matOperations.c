#include "matOperations.h"

void copyMat(float **src, float **dest) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

void resetMat(float **mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            mat[i][j] = 0.0;
        }
    }
}
