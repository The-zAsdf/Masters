#ifndef ERR_H
#define ERR_H

#define ERROR(f) printErr(f)
#define CUDAERROR(err) printCudaErr(err)

void checkCudaSyncErr();
void printErr(int flag);
void printCudaErr(cudaError_t err);

#endif
