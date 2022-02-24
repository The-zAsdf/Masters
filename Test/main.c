#include <stdlib.h>
#include <stdio.h>
#include "IO.h"
#include "PRBM.h"

void main() {
    Var *g = malloc(sizeof(struct Variables));
    double *t;
    readInput("input.txt", g);
    t = malloc(sizeof(double)*g->R);
    printVar(g);

    while (g->index < g->R) {
        t[g->index] = runPRBM(g->W, g->J, g->h, g->N[g->index], g->steps);
        g->index++;
    }
    outputData("time",g->N,t, g->R);

    free(g);
    free(t);
}
