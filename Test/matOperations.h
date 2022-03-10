#ifndef MATOPERATIONS_H
#define MATOPERATIONS_H

#define COPY(src, dest) copyMat(src, dest)
#define RESET(mat) resetMat(mat)

void copyMat(float **src, float **dest);
void resetMat(float **mat);
void setN(int Num);

#endif
