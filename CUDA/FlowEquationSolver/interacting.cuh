#ifndef INTERACTING_CUH
#define INTERACTING_CUH

typedef struct index{
    int i;
    int j;
    int x;
    int y;
} ind;

typedef struct diagElement {
    float el; // element
    float *mel; // matrix element
} diag;

typedef struct offDiagElement {
    float el; // element
    float **mel; // matrix element
} odiag;

typedef struct floet {
    struct diag d;
    struct odiag *o;
    int n;
};

void setVariables(struct Variables *v);
size_t calculateBlocks(size_t threads);
void determineThreadsAndBlocks();
void init();
double runPRBM(struct Variables *v);

#endif
