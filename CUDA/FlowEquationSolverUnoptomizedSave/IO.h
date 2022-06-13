#ifndef IO_H
#define IO_H

/* How to add another variable to input.txt and use it:
 * 1) Add the new variable to the Variables struct below
 * 2) Update IO.cu to read the variable from input.txt (increase argc conditional)
 * 3) Modify FES.py's runFES() function to include the variable (if FES uses var)
 * 4) Let FES.py read in the new variable
 * 5) In interacting.cu: (if FES uses var)
 *    a) init var in __managed__ memory
 *    b) Update setVariables() function
 */

typedef struct Variables{
    double W;
    double J;
    double h;
    int R; // repitition
    int N[10];
    int index;
    double steps;
    double D;
    double etol;
    double cutoff;
} Var;

typedef struct floardH {
    double *h;
    double t;
}fdH;

typedef struct floardD {
    double **D;
    double t;
}fdD;

typedef struct floardG {
    double ****G;
    double t;
}fdG;

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
void outputGRecord(const char *fileName, int n, int r, struct floardG **dR);
void outputH4Record(const char *fileName, int n, int r, struct floardF **dR);
void outputH2Record(const char *fileName, int n, int r, struct floardF **dR);
void outputHamMathematica(const char *fileName, double** mat, double ****ten, int N);
void outputQRecord(const char *fileName, int n, int r, struct floardF **dR);
void outputH4(const char *fileName, double ****mat, int N);
void outputH2(const char *fileName, double **mat, int N);
void outputH2Qu(const char *fileName, double **mat, int N);
void inputH2(const char *fileName, int n, double **mat);
void inputH4(const char *fileName, int n, double ****ten);
void inputh(const char *fileName, int n, double *h);

#endif
