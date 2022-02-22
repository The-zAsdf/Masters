#include <stdio.h>
#include <stdlib.h>
#include "matOperations.h"

float **master;
float **prev;

#define W 10
#define J 10

void init() {
    time_t t;
    srand((unsigned) time(&t));
    master = malloc(sizeof(float*)*N);
    prev = malloc(sizeof(float*)*N);

    for(int i = 0; i < N; i++) {
        master[i] = malloc(sizeof(float)*(N-i));
        prev[i] = malloc(sizeof(float)*(N-i));
        for (int j = 0; j < N-i; j++) {
            if (i == j) {
                master[i][j] = (float)rand()/(float)(RAND_MAX/W);
            } else {
                master[i][j] = (float)rand()/(float)(RAND_MAX/J);
            }
        }
    }
}

float funcJ(int i, int j, float **mat) {
    float m, hi, hj, Jij;
    hi = mat[i][i];
    hj = mat[j][j];
    Jij = mat[i][j];

    m = 0.0;
    for (int k = 0; k < N; k++) {
        m -= mat[i][k]*mat[k][j]*(2.0*mat[k][k]-hi-hj);
    }
    m -= Jij*powf(hi-hj,2.0);

    return m;
}

float funch(int i, float **mat) {
    float m, hi;
    hi = mat[i][i];

    m = 0.0;
    for (int j = 0; j < N; j++) {
        m += powf(mat[i][j],2.0)*(hi-mat[j][j]);
    }
    return 2*m;
}

float nextJ(int i, int j) {

}



void updateMat() {
    copyMat(master, prev);

}
