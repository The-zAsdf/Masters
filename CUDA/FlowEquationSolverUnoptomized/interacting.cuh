#ifndef INTERACTING_CUH
#define INTERACTING_CUH

typedef struct index{
    int i;
    int j;
    int k;
    int l;
} ind;

typedef struct floet {
    float **mat;
    float ****ten;
}f;

void setVariables(struct Variables *v);
size_t calculateBlocks(size_t threads);
void determineThreadsAndBlocks();
void init();
double runPRBM(struct Variables *v);
float readFloet(struct floet *mat, int i, int j, int k, int l);
float findMax(struct floet *mat, int *x, int *y, int *z, int *q);

#endif
