#ifndef IO_H
#define IO_H

typedef struct Variables{
    double W;
    double J;
    double h;
    int R; // repitition
    int N[10];
    int index;
    double steps;
} Var;

typedef struct floardH {
    double *h;
    double t;
}fdH;

typedef struct floardD {
    double **D;
    double t;
}fdD;

typedef struct floardF {
    double f;
    double t;
}fdF;

void readInput(const char *fileName, Var *v);
void readArgs(int argc,char *argv[], Var *v);
void printVar(Var *var);
void outputData (const char *fileName, int *x, double *y, int len);
void outputData (const char *fileName, double y);
void outputHistoryMatrices(const char *fileName, double ***hist, int len, int n);
void outputDiag(const char *filename, double ***hist, int len, int n);
void outputElements(const char *filename, double ***hist, int len, int n);
void printMatrix(double **mat, int n);
void printErrorMatrix(double **mat, int n);
void outputHRecord(const char *fileName, int n, int r, struct floardH **hR);
void outputDRecord(const char *fileName, int n, int r, struct floardD **dR);
void outputiRecord(const char *fileName, int n, int r, struct floardF **dR);
void outputH4(const char *fileName, double ****mat, int N);
void outputH2(const char *fileName, double **mat, int N);
void inputH2(const char *fileName, int n, double **mat);
void inputH4(const char *fileName, int n, double ****ten);

#endif
