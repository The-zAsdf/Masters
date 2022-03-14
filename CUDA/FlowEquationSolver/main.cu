#include <stdlib.h>
#include <stdio.h>
#include "IO.h"
#include "PRBM.h"

int main(int argc, char *argv[]) {
    Var *g = (Var *) malloc(sizeof(struct Variables));
    double *t;

    if (argc == 1) {
        readInput("input.txt", g);
        t = (double *) malloc(sizeof(double)*g->R);
        printVar(g);

        while (g->index < g->R) {
            t[g->index] = runPRBM(g);
            g->index++;
        }
        outputData("time",g->N,t, g->R);
    } else {
        readArgs(argc, argv, g);
        t = (double *) malloc(sizeof(double));
        t[0] = runPRBM(g);
        outputData("time",t[0]);
    }

    free(g);
    free(t);
    return 1;
}
