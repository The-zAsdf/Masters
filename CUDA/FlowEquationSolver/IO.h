#ifndef IO_H
#define IO_H

typedef struct Variables{
    float W;
    float J;
    float h;
    int R; // repitition
    int N[10];
    int index;
    double steps;
} Var;

void readInput(const char *fileName, Var *v);
void readArgs(int argc,char *argv[], Var *v);
void printVar(Var *var);
void outputData (const char *fileName, int *x, double *y, int len);
void outputData (const char *fileName, double y);
void outputHistoryMatrices(const char *fileName, float ***hist, int len, int n);
void outputDiag(const char *filename, float ***hist, int len, int n);
void outputElements(const char *filename, float ***hist, int len, int n);
void printMatrix(float **mat, int n);
void printErrorMatrix(float **mat, int n);

#endif
