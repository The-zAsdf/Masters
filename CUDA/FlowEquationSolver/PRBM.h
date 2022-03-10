#ifndef PRBM_CUH
#define PRBM_CUH

typedef struct index{
    int x;
    int y;
} ind;

void setVariables(Variables *v);
size_t calculateBlocks(size_t threads);
void determineThreadsAndBlocks();
void init();
double runPRBM(Variables *v);

#endif
