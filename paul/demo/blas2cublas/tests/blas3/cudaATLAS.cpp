#include "cublas.h"
#include <stdlib.h>
#include <stdio.h>
#include "cblas.h"
#define n 128

void initMatrix(float *data,int size)
{
  int i;
  for (i = 0; i < size; i++) 
    data[i] = 1;
}


int main(int argc,char *argv[])
{
  //Allocate memory
  float *a = (float *)(malloc((((sizeof(float )) * n) * n)));
  float *b = (float *)(malloc((((sizeof(float )) * n) * n)));
  float *c = (float *)(malloc((((sizeof(float )) * n) * n)));
  float *d1 = (float *)(malloc((((sizeof(float )) * n) * n)));

  //Initialize matrices
  initMatrix(a,(n * n));
  initMatrix(b,(n * n));
  int bi = 0;
  int ci = 0;

  
	float *device1_A;
	float *device1_B;
	float *device1_C;
	int sizeType_device1 = sizeof(float);

	/* Allocate device memory */
	cublasAlloc(n * n, sizeType_device1, (void **)&device1_A);
	cublasAlloc(n * n, sizeType_device1, (void **)&device1_B);
	cublasAlloc(n * n, sizeType_device1, (void **)&device1_C);

	/* Copy matrices to device */
	cublasSetMatrix(n, n, sizeType_device1, (void *)(a + 0), n,
			(void *)device1_A, n);
	cublasSetMatrix(n, n, sizeType_device1, (void *)(b + (bi + ci)), n,
			(void *)device1_B, n);

	/* CUBLAS call */

	cublasSgemm('N', 'N', n, n, n, 1.0, device1_A, n, device1_B,
		    n, 1.0, device1_C, n);

	/* Copy result array back to host */
	cublasSetMatrix(n, n, sizeType_device1, (void *)device1_C, n,
			(void *)c, n);

	/* Free device memory */
	cublasFree(device1_A);
	cublasFree(device1_B);
	cublasFree(device1_C);


  /* Regular BLAS call */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    d1,n, &b[0],n, 1.0, &c[0],n);

  free(a);
  free(b);
  free(c);
  free(d1);
  return 0;
}
