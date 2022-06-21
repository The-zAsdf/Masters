#ifndef INTERACTING_CUH
#define INTERACTING_CUH

// #define SUPPRESSOUTPUT

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
void determineThreadsAndBlocks(size_t *nob, size_t *tpb, int num);
void initFloet(struct floet *mat);
void init();
double runFES(struct Variables *v);
double readFloet(struct floet *mat, int i, int j, int k, int l);
double calcInvariant();
double findMax(struct floet *mat, int *x, int *y, int *z, int *q);
void correctScale(double *scale, double prev, double dprev);
void updateScaleHistory(double **hist, double histlen, double scale);
void copyToRecords(struct floet *mat, double t, int index);
void printH4(struct floet *mat);
void printH4interact(struct floet *mat);
void checkHerm(struct floet *mat);
void checkAHerm(struct floet *mat);
int TESTCONDITION(struct floet *mat);

#endif
