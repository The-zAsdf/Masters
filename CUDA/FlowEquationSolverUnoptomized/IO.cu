#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
                    v->steps = atof(token);
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

void readArgs(int argc, char *argv[], Var *v) {
    if (argc != 6) {
        fprintf(stderr, "Error: Incorrect number of arguments\n");
        fprintf(stderr, "Usage: main <W> <J> <h> <S> <N>\n");
        exit(1);
    }
    v->W = atof(argv[1]);
    v->J = atof(argv[2]);
    v->h = atof(argv[3]);
    v->R = 1;
    v->steps = atof(argv[4]);
    v->N[0] = atoi(argv[5]);
    v->index = 0;
}

void outputData (const char *fileName, int *x, double *y, int len) {
    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    for (int i = 0; i < len; i++) {
        fprintf(fp,"%d,%f\n", x[i], y[i]);
    }
    fclose(fp);
}

void outputData (const char *fileName, double y) {
    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    fprintf(fp,"%f\n", y);
    fclose(fp);
}

void outputHistoryMatrices(const char *fileName, float ***hist, int len, int n) {
    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";
    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fprintf(fp,"%f",hist[i][min(j,k)][abs(j-k)]);
                if (k != n-1) fprintf(fp,",");
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
}

void outputDiag(const char *fileName, float ***hist, int len, int n) {
    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";
    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < len; j++) {
            fprintf(fp,"%f",hist[j][i][0]);
            if (j < len -1) fprintf(fp,",");
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

void outputElements(const char *fileName, float ***hist, int len, int n) {
    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n-i; j++) {
            fprintf(fp,"{%d,%d},",i,j);
            for (int k = 0; k < len; k++) {
                fprintf(fp,"%f",hist[k][i][j]);
                if (k != n-1) fprintf(fp,",");
            }
            fprintf(fp,"\n");
        }
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
    printf("steps: %.2f\n", var->steps);
}

void printMatrix(float **mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f",mat[i][j]);
            if (j != n-1) printf(", ");
        }
        printf("\n");
    }
}

void printErrorMatrix(float **mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.6f",mat[min(i,j)][abs(i-j)]);
            if (j != n-1) printf(", ");
        }
        printf("\n");
    }
}

void outputHRecord(const char *fileName, int n, int r, struct floardH **hR) {

    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    fprintf(fp,"%d,%d,%s,%s\n", r, n, "time", "val");

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp,"%d,%d,%.3f,%.3f\n", i, j, hR[i]->t, hR[i]->h[j]);
        }
    }
}

void outputDRecord(const char *fileName, int n, int r, struct floardD **dR) {

    FILE *fp;
    char str[255];
    const char *dir = "data/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"w+")) == NULL) {
        fprintf(stderr, "Error: File cannot be created\n");
        exit(-1);
    }

    fprintf(fp,"%d,%d,%s,%s\n", r, n, "time", "val");

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                fprintf(fp,"%d,%d,%d,%.3f,%.3f\n", i, j, k, dR[i]->t, dR[i]->D[j][k]);
            }
        }
    }
}
