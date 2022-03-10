#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "matOperations.h"
#include "measureTime.h"

float **master;
float **prev;
float ***kMat;

float W;
float J;
int N;
float h;
int steps;

void init() {
    time_t t;
    srand((unsigned) time(&t));
    master = malloc(sizeof(float*)*N);
    prev = malloc(sizeof(float*)*N);
    kMat = malloc(sizeof(float*)*4);

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

    for (int i = 0; i < 4; i++) {
        kMat[i] = malloc(sizeof(float*)*N);
        for (int j = 0; j < N; j++) {
            kMat[i][j] = malloc(sizeof(float*)*(N-i));
        }
    }

    resetMat(prev);
}


float funcJ(int i, int j, float **mat) {
    float m, hi, hj, Jij;
    hi = mat[i][0];
    hj = mat[j][0];
    Jij = mat[i][j];

    m = 0.0;
    for (int k = 0; k < N; k++) {
        m -= mat[i][k]*mat[k][j]*(2.0*mat[k][0]-hi-hj);
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

void updateMat() {
    float h = 0.5;
    float l = 0.0;
    float ln = l+2.0*h;
    copyMat(master, prev);

    while (l < ln) {
        // Calculating all k1
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-i; j++) {
                if (i == j) {
                    kMat[0][i][j] = funch(i, master);
                } else {
                    kMat[0][i][j] = funcJ(i, j, master);
                }
            }
        }

        // Calculating all k2
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-i; j++) {
                if (i == j) {
                    master[i][j] += h*kMat[0][i][j]/2.0;
                    kMat[1][i][j] = funch(i, master);
                } else {
                    master[i][j] += h*kMat[0][i][j]/2.0;
                    kMat[1][i][j] = funcJ(i, j, master);
                }
            }
        }

        copyMat(prev, master);
        // Calculating all k3
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-i; j++) {
                if (i == j) {
                    master[i][j] += h*kMat[1][i][j]/2.0;
                    kMat[2][i][j] = funch(i, master);
                } else {
                    master[i][j] += h*kMat[1][i][j]/2.0;
                    kMat[2][i][j] = funcJ(i, j, master);
                }
            }
        }

        copyMat(prev, master);
        // Calculating all k4
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-i; j++) {
                if (i == j) {
                    master[i][j] += h*kMat[2][i][j];
                    kMat[3][i][j] = funch(i, master);
                } else {
                    master[i][j] += h*kMat[2][i][j];
                    kMat[3][i][j] = funcJ(i, j, master);
                }
            }
        }

        copyMat(prev, master);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N-i; j++) {
                master[i][j] += h*(kMat[0][i][j] + 2*kMat[1][i][j] + 2*kMat[2][i][j] + kMat[3][i][j])/6.0;
            }
        }

        l += h;
    }
}

double runPRBM(float w, float j, float hh, int n, int ss) {
    W = w;
    J = j;
    h = hh;
    N = n;
    steps = ss;

    init();
    printf("Starting...");
    startTime();
    for (int s = 0; s < steps; s++) { updateMat(); }
    endTime();
    printf("Done\n");
    return runTime();

}
