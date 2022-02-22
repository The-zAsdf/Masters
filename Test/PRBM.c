#include<stdio.h>
#include<math.h>

float **mat;

#define N 100
#define W 10
#define J 10

void init() {
    mat = malloc(sizeof(float*)*N);

    for(int i = 0; i < N; i++) {
        mat[i] = malloc(sizeof(float)*(N-i));
        for (int j = 0; j < N-i; j++) {
            if (i == j) {
                mat[i][j] = (float)rand()/(float)(RAND_MAX/W);
            } else {
                mat[i][j] = (float)rand()/(float)(RAND_MAX/J);
            }
        }
    }
}
