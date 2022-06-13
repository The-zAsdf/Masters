#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <math.h>
#include <stdlib.h>
#include "erfinv.h"
#include "err.cuh"

// Gaussian inverse CDF
double gaussianICDF(double p, double d, double j, double alpha) {
    return (double) (my_erfinvf((float) (2.0f*p-1.0f)))*j*sqrt(2.0)/pow(d,alpha);
}

#endif
