#ifndef PRBM_H
#define PRBM_H

typedef struct index{
    int x;
    int y;
} ind;

void setVariables(struct Variables *v);
size_t calculateBlocks(size_t threads);
void determineThreadsAndBlocks();
void init();
double runPRBM(struct Variables *v);

#endif
