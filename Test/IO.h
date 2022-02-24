#ifndef IO_H
#define IO_H

typedef struct Variables{
    float W;
    float J;
    float h;
    int R; // repitition
    int N[10];
    int index;
    int steps;
} Var;

void readInput(const char *fileName, Var *v);
void printVar(Var *var);
void outputData (const char *fileName, int *x, double *y, int len);

#endif
