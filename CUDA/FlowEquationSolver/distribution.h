#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

float gaussianICDF(float p, float d);
void gererateSUD(int size, float j, float a);
void generateICDF();
float getSampleNumber(int i, int j);
void freeDistributions();

#endif
