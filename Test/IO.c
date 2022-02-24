#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "IO.h"

void readInput(const char *fileName, Var *v) {
    FILE *fp;
    char buff[255];
    char *token;
    if ((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error: No filename found\n");
        exit(1);
    } else {
        while (fgets(buff, 255, fp) != NULL) {
            switch (buff[0]) {
                case 'W':
                    token = strtok(buff, " ");
                    v->W = atof(strtok(NULL, " "));
                    break;
                case 'J':
                    token = strtok(buff, " ");
                    v->J = atof(strtok(NULL, " "));
                    break;
                case 'h':
                    token = strtok(buff, " ");
                    v->h = atof(strtok(NULL, " "));
                    break;
                case 'R':
                    token = strtok(buff, " ");
                    v->R = atoi(strtok(NULL, " "));
                    break;
                case 'S':
                    token = strtok(buff, " ");
                    v->steps = atoi(strtok(NULL, " "));
                    break;
                case 'N':
                    token = strtok(buff, " ");
                    for (int i = 0; i < v->R; i++) {
                        v->N[i] = atoi(strtok(NULL, " "));
                    }
                    break;
            }
        }

        v->index = 0;
        fclose(fp);
    }
}

void outputData (const char *fileName, int *x, double *y, int len) {
    FILE *fp;
    int l;
    char str[255];
    char *dir = "data/";
    char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    printf("%s", str);

    fp = fopen(str,"w+");

    for (int i = 0; i < len; i++) {
        printf("x: %d\t y: %f\n", x[i], y[i]);
        fprintf(fp, "%d,%f\n", x[i], y[i]);
        fflush(fp);
    }
    fclose(fp);
}

void printVar(Var *var) {
    printf("W: %f\n", var->W);
    printf("J: %f\n", var->J);
    printf("h: %f\n", var->h);
    printf("R: %f\n", var->R);
    printf("N:");
    for (int i = 0; i < var->R; i++) { printf(" %d", var->N[i]); }
    printf("\n");
    printf("steps: %d", var->steps);
}
