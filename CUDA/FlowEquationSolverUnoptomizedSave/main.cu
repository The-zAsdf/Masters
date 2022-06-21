#include <stdlib.h>
#include <stdio.h>
#include "IO.h"
#include "interacting.cuh"

int main(int argc, char *argv[]) {
    Var *g = (Var *) malloc(sizeof(struct Variables));
    double *t;

    if (argc == 1) {
        readInput("input.txt", g);
        t = (double *) malloc(sizeof(double)*g->R);
        // #ifndef SUPPRESSOUTPUT
        printVar(g);
        // #endif

        while (g->index < g->R) {
            t[g->index] = runFES(g);
            g->index++;
        }
        outputData("time",g->N,t, g->R);
    } else {
        readArgs(argc, argv, g);
        printVar(g);
        t = (double *) malloc(sizeof(double));
        t[0] = runFES(g);
        outputData("time",t[0]);
    }

    free(g);
    free(t);
    return 0;
}
