
#include <stdio.h>
#include "cblas.h"

void print_array(int n, double *A) {
  int i;
  for (i = 0; i < n; i++) {
    printf("%4.2lf ", A[i]);
  }
  printf("\n");
}

int main (void) {

  int n = 6;
  double a = 2.0;
  double X[] = { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 };
  double Y[] = { 11.0, 12.0, 13.0, 21.0, 22.0, 23.0 };
  
  printf("1. X: "); print_array(n, X);
  printf("1. Y: "); print_array(n, Y);

  // compute Y = aX + Y
  cblas_daxpy(n, a, X, 1, Y, 1);

  printf("2. Y: "); print_array(n, Y);
  
  return 0;
}


