#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "matOperations.h"
#include "measureTime.h"
#include "IO.h"
#include "err.h"

#define SAVES 100

float **master;
float **temp;
float ***history;
float ***kMat;

float W;
float J;
int N;
float h;
int steps;

void setVariables(Var *v) {
    W = v->W;
    J = v->J;
    N = v->N[v->index];
    h = v->h;
    steps = v->steps;
    setN(N);
}

void init() {
    time_t t;

    srand((unsigned) time(&t));
    if ((master = malloc(sizeof(float*)*N)) == NULL) { ERROR(1); }
    if ((temp = malloc(sizeof(float*)*N)) == NULL) { ERROR(1); }
    if ((history = malloc(sizeof(float*)*SAVES)) == NULL) { ERROR(1); }
    if ((kMat = malloc(sizeof(float*)*4)) == NULL) { ERROR(1); }

    // master
    for (int i = 0; i < N; i++){
        if ((master[i] = malloc(sizeof(float)*(N-i))) == NULL) { ERROR(1); }
        if ((temp[i] = malloc(sizeof(float)*(N-i))) == NULL) { ERROR(1); }
        for (int j = 0; j < N-i; j++) {
            // init values of master
        }
    }

    // history
    for (int i = 0; i < SAVES; i++) {
        if ((history[i] = malloc(sizeof(float*)*N);
        for (int j = 0; j < N; j++) {
            if ((history[i][j] = malloc(sizeof(float)*(N-j))) == NULL) { ERROR(1); }
        }
    }

    // kMat
    for (int i = 0; i < 4; i++) {
        if ((kMat[i] = malloc(sizeof(float*)*N);
        for (int j = 0; j < N; j++) {
            if ((kMat[i][j] = malloc(sizeof(float)*(N-j)) == NULL) { ERROR(1); }
        }
    }
}

void freeMem() {
    for (int i = 0; i < N; i++) {
        free(master[i]);
        free(temp[i]);
    }
    free(master);
    free(temp);

    for (int i = 0; i < SAVES; i++) {
        for (int j = 0; j < N; j++) {
            free(history[i][j]);
        }
        free(history[i]);
    }
    free(history);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < N; j++) {
            free(kMat[i][j]);
        }
        free(kMat[i]);
    }
    free(kMat);
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

void adaptiveScaling(float h) {
    return h; // TBC; for later
}

void updateMat() {
    float h = 0.1; // Initial value; to be changed; higher = less accuracy
}

double runPRBM(Var *v) {
    setVariables(v);
    init();
    startTime();
    for (int s = 0; s < steps; s++) { updateMat(); }
    endTime();
    freeMem();
    return runTime();
}
