#include <stdlib.h>
#include <stdio.h>
#include "mkl.h"

#define n 8192


void initMatrix(float* data, int size){
  int i;
  for( i=0;i<size;i++)
    data[i] = 1;
}

//Program main
int main(int argc, char* argv[])
{
	//Allocate memory
	float *a = (float*) malloc(sizeof(float) * n * n);
	float *b = (float*) malloc(sizeof(float) * n * n);
	float *c = (float*) malloc(sizeof(float) * n * n);
	float *d1 = (float*) malloc(sizeof(float) * n * n);

	//Initialize matrices
	initMatrix(a,n*n);
	initMatrix(b,n*n);
        int bi = 0,ci=0;

	//BLAS call
        // %BLAS2CUBLAS prefix=device1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    &a[0],n, &b[bi+ci],n, 1.0, c,n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    d1,n, &b[0],n, 1.0, &c[0],n);

	free(a);
	free(b);
	free(c);

	return 0;
}
        


