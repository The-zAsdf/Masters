#include "main.cuh"

__managed__ struct floet **master;
__managed__ struct floet **prev;
__managed__ struct floet **temp;
__managed__ struct floet ***kMat;

int main() {
    size_t bytes = 0;
    cudaError_t err;
    int N = 20;
    int numElem = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N-i; j++) {
            numElem++;
            if (j == 0) { // Diag elements
                for (int x = 0; x < N; x++) {
                    numElem++;
                }
            } else { // Off diag elements
                for (int x = 0; x < N; x++) {
                    for (int y = 0; y < N; y++) {
                        numElem++;
                    }
                }
            }
        }
    }

    err = cudaMallocManaged(&master, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    bytes += sizeof(struct floet**)*N;

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&master[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(struct floet*);

        err = cudaMallocManaged(&master[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*N;

        err = cudaMallocManaged(&master[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*(N-i);

        err = cudaMallocManaged(&master[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float *)*(N-i);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&master[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float *)*N;
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&master[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
                bytes += sizeof(float)*N;
            }
        }
    }

    err = cudaMallocManaged(&prev, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    bytes += sizeof(struct floet**)*N;

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&prev[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(struct floet*);

        err = cudaMallocManaged(&prev[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*N;

        err = cudaMallocManaged(&prev[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*(N-i);

        err = cudaMallocManaged(&prev[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float *)*(N-i);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&prev[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float *)*N;
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&prev[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
                bytes += sizeof(float)*N;
            }
        }
    }

    err = cudaMallocManaged(&temp, sizeof(struct floet**)*N);
    if (err != cudaSuccess) CUDAERROR(err);
    bytes += sizeof(struct floet**)*N;

    for (int i = 0; i < N; i++) {
        err = cudaMallocManaged(&temp[i], sizeof(struct floet*));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(struct floet*);

        err = cudaMallocManaged(&temp[i]->d, sizeof(float)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*N;

        err = cudaMallocManaged(&temp[i]->j, sizeof(float)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float)*(N-i);

        err = cudaMallocManaged(&temp[i]->g, sizeof(float *)*(N-i));
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(float *)*(N-i);

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&temp[i]->g[j], sizeof(float *)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float *)*N;
            for(int k = 0; k < N; k++) {
                err = cudaMallocManaged(&temp[i]->g[j][k], sizeof(float)*N);
                if (err != cudaSuccess) CUDAERROR(err);
                bytes += sizeof(float)*N;
            }
        }
    }

    err = cudaMallocManaged(&kMat, sizeof(struct floet***)*7);
    if (err != cudaSuccess) CUDAERROR(err);
    bytes += sizeof(struct floet***)*7;

    for (int i = 0; i < 7; i++) {
        err = cudaMallocManaged(&kMat[i], sizeof(struct floet**)*N);
        if (err != cudaSuccess) CUDAERROR(err);
        bytes += sizeof(struct floet**)*N;

        for (int j = 0; j < N; j++) {
            err = cudaMallocManaged(&kMat[i][j], sizeof(struct floet*));
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(struct floet*);

            err = cudaMallocManaged(&kMat[i][j]->d, sizeof(float)*N);
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float)*N;

            err = cudaMallocManaged(&kMat[i][j]->j, sizeof(float)*(N-i));
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float)*(N-i);

            err = cudaMallocManaged(&kMat[i][j]->g, sizeof(float *)*(N-i));
            if (err != cudaSuccess) CUDAERROR(err);
            bytes += sizeof(float *)*(N-i);

            for (int k = 0; k < N; k++) {
                err = cudaMallocManaged(&kMat[i][j]->g[k], sizeof(float *)*N);
                if (err != cudaSuccess) CUDAERROR(err);
                bytes += sizeof(float *)*N;
                for (int l = 0; l < N; l++) {
                    err = cudaMallocManaged(&kMat[i][j]->g[k][l], sizeof(float)*N);
                    if (err != cudaSuccess) CUDAERROR(err);
                    bytes += sizeof(float)*N;
                }
            }
        }
    }

    printf("sizeof(all) = %zd bytes (%f MB)\n",bytes, (double) bytes/1024.0/1024.0);

    return 0;
}
