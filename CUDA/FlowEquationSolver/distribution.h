#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

float gaussianICDF(float p, float d);
void generateSUD(float a, float j, float w, int num);
void generateICDF();
void freeDistributions();
__global__ void generateMaster(curandState_t* states, float** master);

#endif
