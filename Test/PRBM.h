#ifndef PRBM_H
#define PRBM_H
#include "matOperations.h"
#include "measureTime.h"
#include "IO.h"

float **master;
float **prev;
float ***kMat;
float W;
float J;
int N;
float h;
int steps;

void init();
float funcJ(int i, int j, float **mat);
float funch(int i, float **mat);
void updateMat();
double run();

#endif
