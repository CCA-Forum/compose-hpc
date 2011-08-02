
#include <stdio.h>
#include "distarray_BlockDistArray2dInt.h"

void client_main(void) {
  printf(" distarray_BlockDistArray2dInt_cClient.client_main(): Starting...\n");

  sidl_BaseInterface ex;
  distarray_BlockDistArray2dInt distArray = hplsupport_BlockCyclicDistArray2dDouble__create(&ex);

  distarray_BlockDistArray2dInt_initArray(distArray, 0, 3, 0, 3, 2, 2, &ex);

  int32_t dim = distarray_BlockDistArray2dInt_getDimension(distArray, &ex);
  printf(" distarray_BlockDistArray2dInt_cClient.client_main(): dim = %d \n", dim);

  int32_t lo2 = distarray_BlockDistArray2dInt_getLower(distArray, 1, &ex);
  printf(" distarray_BlockDistArray2dInt_cClient.client_main(): lo2 = %d \n", lo2);

  int32_t hi2 = distarray_BlockDistArray2dInt_getHigher(distArray, 1, &ex);
  printf(" distarray_BlockDistArray2dInt_cClient.client_main(): hi2 = %d \n", hi2);

  for (int i = 0; i <= 3; i++) {
	for (int j = 0; j <= 3; j++) {
	  distarray_BlockDistArray2dInt_setIntoArray(distArray, 10 * i + j, i, j, &ex);
	}
  }

  for (int i = 0; i <= 3; i++) {
  	for (int j = 0; j <= 3; j++) {
  	  int32_t val = distarray_BlockDistArray2dInt_getFromArray(distArray, i, j, &ex);
  	  printf("%d ", val);
  	}
  	printf("\n");
  }


  distarray_BlockDistArray2dInt A = distarray_BlockDistArray2dInt__create(&ex);
  distarray_BlockDistArray2dInt_initArray(A, 0, 3, 0, 3, 2, 2, &ex);
  distarray_BlockDistArray2dInt B = distarray_BlockDistArray2dInt__create(&ex);
  distarray_BlockDistArray2dInt_initArray(B, 0, 3, 0, 3, 2, 2, &ex);
  distarray_BlockDistArray2dInt C = distarray_BlockDistArray2dInt__create(&ex);
  distarray_BlockDistArray2dInt_initArray(C, 0, 3, 0, 3, 2, 2, &ex);
  for (int i = 0; i <= 3; i++) {
    for (int j = 0; j <= 3; j++) {
      distarray_BlockDistArray2dInt_setIntoArray(A, 10 * i + j, i, j, &ex);
      distarray_BlockDistArray2dInt_setIntoArray(B, i + 10 * j, i, j, &ex);
      distarray_BlockDistArray2dInt_setIntoArray(C, 0, i, j, &ex);
    }
  }
  distarray_BlockDistArray2dInt_matrixMultipleCannon(&A, &B, &C, &ex);

  for (int i = 0; i <= 3; i++) {
    for (int j = 0; j <= 3; j++) {
      int32_t val = distarray_BlockDistArray2dInt_getFromArray(C, i, j, &ex);
      printf("C[ %d, %d] = %d ", i, j, val);
    }
  }

  printf(" distarray_BlockDistArray2dInt_cClient.client_main(): Ending.\n");
}
