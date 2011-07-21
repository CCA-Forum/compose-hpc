#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "cblas.h"

#ifdef __cplusplus
}
#endif /* __cplusplus */

#define n 1024


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

	//BLAS calls

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    &a[0],n, &b[bi+ci],n, 1.0, c,n);


	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    d1,n, &b[0],n, 1.0, &c[0],n);

	free(a);
	free(b);
	free(c);
	free(d1);

	return 0;
}
        


