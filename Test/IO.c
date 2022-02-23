#include <stdio.h>
#include <stdlib.h>
#include "IO.h"

typedef struct Variables{
    float W;
    float J;
    float h;
    int R; // repitition
    int N[10];
    int index = 0;
    int steps;
} var;

void readInput(const char *fileName, var *v) {
    FILE *fp;
    char buff[255];
    if ((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error: No filename found\n");
        exit(1);
    } else {
        while (fscanf(fp, "%s", buff) != EOF) {
            switch (buff[0]) {
                case 'W':
                    if (fscanf(fp, "%f", v->W) != (EOF || 0)) {
                        printf("W: %f\n", v->W);
                    } else {
                        fprintf(stderr, "Error: Input error for W");
                        exit(2);
                    }
                case 'J':
                    if (fscanf(fp, "%f", v->J) != (EOF || 0)) {
                        printf("J: %f\n", v->J);
                    } else {
                        fprintf(stderr, "Error: Input error for J");
                        exit(2);
                    }
                case 'R':
                    if (fscanf(fp, "%d", v->R) != (EOF || 0)) {
                        printf("R: %d\n", v->R);
                    } else {
                        fprintf(stderr, "Error: Input error for R");
                        exit(2);
                    }
                case 'S':
                    if (fscanf(fp, "%d", v->steps) != (EOF || 0)) {
                        printf("steps: %d\n", v->steps);
                    } else {
                        fprintf(stderr, "Error: Input error for steps");
                        exit(2);
                    }
                case 'h':
                    if (fscanf(fp, "%f", v->h) != (EOF || 0)) {
                        printf("steps: %f\n", v->h);
                    } else {
                        fprintf(stderr, "Error: Input error for h");
                        exit(2);
                    }
                case 'N':
                    for (int i = 0; i < v->R; i++) {
                        if (fscanf(fp, "%d", v->N[i]) != (EOF || 0)) {
                            printf("N: %d\n", v->N);
                        } else {
                            fprintf(stderr, "Error: Input error for N");
                            exit(2);
                        }
                    }
            }
        }
        fclose(fp);
    }
}

void setupVar(float *w, float *j, float *h, int *size, int *steps, var *v) {
    *w = v->W;
    *j = v->J;
    *h = v->h;
    *size = v->N[index];
    *steps = v->steps;
}
