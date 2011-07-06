
#include <stdio.h>
#include <math.h>
#include "hplsupport.h"

#define DEBUG 0

void panelSolve(void* abData, void* pivData, /* abLimits*/ int abStart1, int abEnd1, int abStart2, int abEnd2, /*panel domain*/ int start1, int end1, int start2, int end2) {

  if (DEBUG) printf("panelSolve(%p, %p, /*ab*/ %d, %d, %d, %d, /*panel*/ %d, %d, %d, %d) \n", abData, pivData,
		  abStart1, abEnd1, abStart2, abEnd2,
		  start1, end1, start2, end2);

  sidl_BaseInterface ex;

  hplsupport_BlockCyclicDistArray2dDouble ab = hplsupport_BlockCyclicDistArray2dDouble__create(&ex);
  hplsupport_BlockCyclicDistArray2dDouble_initData(ab, abData, &ex);

  hplsupport_SimpleArray1dInt piv = hplsupport_SimpleArray1dInt__create(&ex);
  hplsupport_SimpleArray1dInt_initData(piv, pivData, &ex);

  // for k in panel.dim(2) {
  for (int k = start2; k <= end2; k++) {

	if (DEBUG) printf(" k = %d \n", k);

	// Find the pivot, the element with the largest absolute value.
	double pivotVal = 0;
	int pivotRow = -1;
	for (int p = k; p <= end1; p++) {
      double loopVal = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, p, k, &ex);
      if (fabs(loopVal) > pivotVal) {
    	pivotVal = loopVal;
    	pivotRow = p;
      }
	}
	if (DEBUG) printf("  pivotRow = %d, pivotVal = %f \n", pivotRow, pivotVal);
	if (pivotRow == -1) {
	  // Nothing to do
	  if (DEBUG) printf("no pivot row found, returning");
	  return;
	}

	// Swap the current row with the pivot row and update the pivot vector to reflect that
	if (pivotRow != k) {
      if (DEBUG) printf("  swapping rows in ab \n");
	  // Ab[k..k, ..] <=> Ab[pivotRow..pivotRow, ..];
	  for (int c = abStart2; c <= abEnd2; c++) {
	    double ab1 = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, k, c, &ex);
        double ab2 = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, pivotRow, c, &ex);
        hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(ab, ab2, k, c, &ex);
        hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(ab, ab1, pivotRow, c, &ex);
	  }
	  if (DEBUG) printf("  swapping rows in piv \n");
	  // piv[k] <=> piv[pivotRow];
	  {
		  int32_t p1 = hplsupport_SimpleArray1dInt_getFromArray(piv, k, &ex);
		  int32_t p2 = hplsupport_SimpleArray1dInt_getFromArray(piv, pivotRow, &ex);
		  hplsupport_SimpleArray1dInt_setIntoArray(piv, p2, k, &ex);
		  hplsupport_SimpleArray1dInt_setIntoArray(piv, p1, pivotRow, &ex);
	  }
	}

	if (pivotVal == 0) {
	  // Matrix cannot be factorized
      if (DEBUG) printf("Matrix cannot be factorized\n");
      return;
	}

	if (DEBUG) printf("  normalizing values of col-%d in ab \n", k);
	// divide all values below and in the same col as the pivot by the pivot value
	for (int r = k + 1; r <= abEnd1; r++) {
	  double ab1 = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, r, k, &ex);
	  double ab2 = ab1 / pivotVal;
	  hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(ab, ab2, r, k, &ex);
	}

	if (DEBUG) printf("  updating remaining values of ab \n", k);
	// update all other values below the pivot
	for (int i = k + 1; i <= end1; i++) {
      for (int j = k + 1; j <= end2; j++) {
    	// Ab[i,j] -= Ab[i,k] * Ab[k,j];
    	double ab_ij = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, i, j, &ex);
    	double ab_ik = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, i, k, &ex);
    	double ab_kj = hplsupport_BlockCyclicDistArray2dDouble_getFromArray(ab, k, j, &ex);
    	double newVal = ab_ij - (ab_ik * ab_kj);
    	hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(ab, newVal, i, j, &ex);
	  }
	}
  }
}

