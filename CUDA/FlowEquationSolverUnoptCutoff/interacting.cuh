#ifndef INTERACTING_CUH
#define INTERACTING_CUH

typedef struct index{
    int i;
    int j;
    int k;
    int l;
} ind;

typedef struct floet {
    double **mat;
    double ****ten;
}f;

void setVariables(struct Variables *v);
size_t calculateBlocks(size_t threads);
void determineThreadsAndBlocks();
void initFloet(struct floet *mat);
void init();
double runPRBM(struct Variables *v);
double readFloet(struct floet *mat, int i, int j, int k, int l);
double calcInvariant();
double findMax(struct floet *mat, int *x, int *y, int *z, int *q);
void copyToRecords(struct floet *mat, double t, int index);
void printH4(struct floet *mat);
void printH4interact(struct floet *mat);
void checkHerm(struct floet *mat);
void checkAHerm(struct floet *mat);
int TESTCONDITION(struct floet *mat);

#endif
