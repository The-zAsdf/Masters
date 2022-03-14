#include <math.h>
#include <stdlib.h>
#include "erfinv.h"
#include "err.h"

// Gaussian inverse CDF
float gaussianICDF(float p, float d, float j, float alpha) {
    return my_erfinvf(2.0*p-1.0)*j*sqrtf(2.0)/powf(d,alpha);
}
