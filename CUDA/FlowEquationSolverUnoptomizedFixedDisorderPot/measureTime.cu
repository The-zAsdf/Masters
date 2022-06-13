#include "measureTime.h"

clock_t start;
clock_t end;

void startTime() {
    start = clock();
}

void endTime() {
    end = clock();
}

double runTime() {
    return (double)(end - start) / CLOCKS_PER_SEC;
}
