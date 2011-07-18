
#ifndef DAXPY_SUPPORT_H
#define DAXPY_SUPPORT_H

#include <stdio.h>

// Macro to return the pointer
#define makeOpaque(aPtr) (aPtr)

void printArray(int len, double* arr) {
  int i;
  for (i = 0; i < len; i++) {
    printf("%4.2lf ", arr[i]);
  }
  printf("\n");
}

#endif