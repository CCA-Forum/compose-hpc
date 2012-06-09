#include <stdlib.h>
#include <stdio.h>
#include "mkl.h"

#define n 100


void initMatrix(float* data, int size){
  int i;
  int c=1;
  for( i=0;i<size;i++){
    data[i] = c;
    c=c+1;
  }
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
        initMatrix(d1,n*n);
        int bi = 0,ci=0,i=0;

	//BLAS call

        // %BLAS2CUBLAS prefix=device1
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    &a[0],n, &b[bi+ci],n, 0.0, c,n);

	// %BLAS2CUBLAS prefix=device2
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    d1,n, &b[0],n, 0.0, &c[0],n);

        for( i=0;i<n*n;i++) printf("%f\n",c[i]);
	    

	free(a);
	free(b);
	free(c);
	free(d1);

	return 0;
}
        


