#ifndef INTERACTING_CUH
#define INTERACTING_CUH

typedef struct index{
    int i;
    int j;
    int k;
    int l;
} ind;

typedef struct floet {
    float h;
    float *d;
    float *j;
    float ***g;
}f;

void setVariables(struct Variables *v);
size_t calculateBlocks(size_t threads);
size_t calculateBlocksHH(size_t threads);
size_t calculateBlocksJJ(size_t threads);
void determineThreadsAndBlocks();
void determineThreadsAndBlocksHH();
void determineThreadsAndBlocksJJ();
void init();
double runPRBM(struct Variables *v);
float readFloet(struct floet **mat, int i, int j, int k, int l);

#endif
