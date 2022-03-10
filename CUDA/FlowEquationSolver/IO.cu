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
            token = strtok(buff, " ");
            switch (buff[0]) {
                case 'W':
                    token = strtok(NULL, " ");
                    v->W = atof(token);
                    break;
                case 'J':
                    token = strtok(NULL, " ");
                    v->J = atof(token);
                    break;
                case 'h':
                    token = strtok(NULL, " ");
                    v->h = atof(token);
                    break;
                case 'R':
                    token = strtok(NULL, " ");
                    v->R = atoi(token);
                    break;
                case 'S':
                    token = strtok(NULL, " ");
                    v->steps = atoi(token);
                    break;
                case 'N':
                    for (int i = 0; i < v->R; i++) {
                        token = strtok(NULL, " ");
                        v->N[i] = atoi(token);
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
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    printf("%s\n", str);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(1);
    }

    for (int i = 0; i < len; i++) {
        fprintf(fp,"%d,%f\n", x[i], y[i]);
    }
    fclose(fp);
}

void printVar(Var *var) {
    printf("W: %f\n", var->W);
    printf("J: %f\n", var->J);
    printf("h: %f\n", var->h);
    printf("R: %d\n", var->R);
    printf("N:");
    for (int i = 0; i < var->R; i++) { printf(" %d", var->N[i]); }
    printf("\n");
    printf("steps: %d\n", var->steps);
}
