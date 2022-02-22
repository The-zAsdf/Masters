#include <stdio.h>
#include <stdlib.h>
#define SIZE	1024

void VectorAdd(int *a, int *b, int *c, int n)
{
	int i;

	for (i=0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	VectorAdd(a, b, c, SIZE);

	for (int i = 0; i < 10; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
