#include <math.h>
#include <stdlib.h>
#include "erfinv.h"

float *uniform;
float **invGaus;
int s;

// Gaussian inverse CDF
float gaussianICDF(float p, float d) {
    return my_erfinvf(2.0*p-1.0)*j0*sqrtf(2.0)/powf(d,alpha);
}

// Generate standard uniform distribution
void gererateSUD(int size) {
    uniform = malloc(sizeof(float)*size);
    s = size;

    for (int i = 0; i < s; i++) {
        uniform[i] = (float) i/(float) (s-1);
    }
}

// Generate ICDF values
void generateICDF(int n) {
    int r;
    invGaus = malloc(sizeof(float*)*n);
    for (int i = 0; i < n; i++) {
        invGaus[i] = malloc(sizeof(float)*s);
        for (int j = 0; j < s; j++) {
            r = rand()%s;
            invGaus[i][j] = gaussianICDF(uniform[r], i);
        }
    }
}

float getSampleNumber(int i, int j) {
    int r = rand()%s;

    return invGaus[abs(i-j)][r];
}
