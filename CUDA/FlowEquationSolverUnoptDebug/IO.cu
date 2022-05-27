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
                case 'e':
                    token = strtok(NULL, " ");
                    v->etol = atof(token);
                    break;
                case 'D':
                    token = strtok(NULL, " ");
                    v->D = atof(token);
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

void outputHistoryMatrices(const char *fileName, double ***hist, int len, int n) {
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

void outputDiag(const char *fileName, double ***hist, int len, int n) {
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

void outputElements(const char *fileName, double ***hist, int len, int n) {
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
    printf("etol: %.10f\n", var->etol);
    printf("D: %.5f\n", var->D);
}

void outputH2(const char *fileName, double** mat, int N) {
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

    fprintf(fp,"{");
    for (int i = 0; i < N; i++) {
        fprintf(fp,"{");
        for (int j = 0; j < N; j++) {
            fprintf(fp,"%.5f", mat[i][j]);
            if (j < N-1) fprintf(fp,",");
        }
        fprintf(fp,"}");
        if (i < N-1) fprintf(fp,",");
    }
    fprintf(fp,"}");
    fclose(fp);
}

void outputH4(const char *fileName, double ****mat, int N) {
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

    fprintf(fp,"{");
    for (int i = 0; i < N; i++) {
        fprintf(fp,"{");
        for (int j = 0; j < N; j++) {
            fprintf(fp,"{");
            for (int k = 0; k < N; k++) {
                fprintf(fp,"{");
                for (int l = 0; l < N; l++) {
                    fprintf(fp,"%.5f", mat[i][j][k][l]);
                    if (l < N-1) fprintf(fp,",");
                }
                fprintf(fp,"}");
                if (k < N-1) fprintf(fp,",");
            }
            fprintf(fp,"}");
            if (j < N-1) fprintf(fp,",");
        }
        fprintf(fp,"}");
        if (i < N-1) fprintf(fp,",");
    }
    fprintf(fp,"}");
    fclose(fp);
}

void printMatrix(double **mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f",mat[i][j]);
            if (j != n-1) printf(", ");
        }
        printf("\n");
    }
}

void printErrorMatrix(double **mat, int n) {
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
    fclose(fp);
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
    fclose(fp);
}

void outputiRecord(const char *fileName, int n, int r, struct floardF **dR) {

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

    fprintf(fp,"%d,%d,%s,%s\n", r, n, "time", "invariant");

    for (int i = 0; i < r; i++) {
        fprintf(fp,"%d,%.3f,%.3f\n", i, dR[i]->t, dR[i]->f);
    }
    fclose(fp);
}

void inputH2(const char *fileName, int n, double **mat) {
    FILE *fp;
    char str[255];
    const char *dir = "input/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"r")) == NULL) {
        fprintf(stderr, "Error: File cannot be read\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(fp,"%lf\n ",&mat[i][j]) != 1) {
                printf("ERROR: Reading in H2 (%d, %d)\n", i, j);
                exit(-1);
            }
        }
    }
    fclose(fp);
}

void inputH4(const char *fileName, int n, double ****ten) {
    FILE *fp;
    char str[255];
    const char *dir = "input/";
    const char *ext = ".txt";

    strcpy(str, dir);
    strcat(str, fileName);
    strcat(str, ext);

    if ((fp = fopen(str,"r")) == NULL) {
        fprintf(stderr, "Error: File cannot be read\n");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    if (fscanf(fp,"%lf\n",&ten[i][j][k][l]) != 1) {
                        printf("ERROR: Reading in H4\n");
                        exit(-1);
                    }
                }
            }
        }
    }
    fclose(fp);
}
