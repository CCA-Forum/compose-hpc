/*
 * File:          hpcc_HighPerformanceLinpack_Impl.c
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hpcc.HighPerformanceLinpack" (version 0.1)
 */

#include "hpcc_HighPerformanceLinpack_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._includes) */
#include <stdio.h>
#include <math.h>
#include "hplsupport.h"
#define DEBUG 0
/* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__ctor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._ctor) */
    /*
     * // boilerplate constructor
     * struct hpcc_HighPerformanceLinpack__data *dptr = (struct hpcc_HighPerformanceLinpack__data*)malloc(sizeof(struct hpcc_HighPerformanceLinpack__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct hpcc_HighPerformanceLinpack__data));
     *   // initialize elements of dptr here
     * hpcc_HighPerformanceLinpack__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hpcc.HighPerformanceLinpack._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__ctor2(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__dtor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._dtor) */
    /*
     * // boilerplate destructor
     * struct hpcc_HighPerformanceLinpack__data *dptr = hpcc_HighPerformanceLinpack__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   hpcc_HighPerformanceLinpack__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._dtor) */
  }
}

/*
 * Method:  panelSolveCompute[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack_panelSolveCompute"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack_panelSolveCompute(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble ab,
  /* in */ hplsupport_SimpleArray1dInt piv,
  /* in */ int32_t abStart1,
  /* in */ int32_t abEnd1,
  /* in */ int32_t abStart2,
  /* in */ int32_t abEnd2,
  /* in */ int32_t start1,
  /* in */ int32_t end1,
  /* in */ int32_t start2,
  /* in */ int32_t end2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack.panelSolveCompute) */
    if (DEBUG) printf("impl_hpcc_HighPerformanceLinpack_panelSolveCompute(%p, %p, /*ab*/ %d, %d, %d, %d, /*panel*/ %d, %d, %d, %d) \n", ab, piv,
          abStart1, abEnd1, abStart2, abEnd2,
          start1, end1, start2, end2);

    sidl_BaseInterface ex;

    // for k in panel.dim(2) {
    for (int k = start2; k <= end2; k++) {

    if (DEBUG) printf(" k = %d \n", k);

    // Find the pivot, the element with the largest absolute value.
    double pivotVal = 0;
    int pivotRow = -1;
    for (int p = k; p <= end1; p++) {
      double loopVal = hplsupport_BlockCyclicDistArray2dDouble_get(ab, p, k, &ex);
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
          double ab1 = hplsupport_BlockCyclicDistArray2dDouble_get(ab, k, c, &ex);
          double ab2 = hplsupport_BlockCyclicDistArray2dDouble_get(ab, pivotRow, c, &ex);
          hplsupport_BlockCyclicDistArray2dDouble_set(ab, ab2, k, c, &ex);
          hplsupport_BlockCyclicDistArray2dDouble_set(ab, ab1, pivotRow, c, &ex);
        }
        if (DEBUG) printf("  swapping rows in piv \n");
        // piv[k] <=> piv[pivotRow];
        {
          int32_t p1 = hplsupport_SimpleArray1dInt_get(piv, k, &ex);
          int32_t p2 = hplsupport_SimpleArray1dInt_get(piv, pivotRow, &ex);
          hplsupport_SimpleArray1dInt_set(piv, p2, k, &ex);
          hplsupport_SimpleArray1dInt_set(piv, p1, pivotRow, &ex);
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
        double ab1 = hplsupport_BlockCyclicDistArray2dDouble_get(ab, r, k, &ex);
        double ab2 = ab1 / pivotVal;
        hplsupport_BlockCyclicDistArray2dDouble_set(ab, ab2, r, k, &ex);
      }

      if (DEBUG) printf("  updating remaining values of ab \n");
      // update all other values below the pivot
      for (int i = k + 1; i <= end1; i++) {
        for (int j = k + 1; j <= end2; j++) {
          // Ab[i,j] -= Ab[i,k] * Ab[k,j];
          double ab_ij = hplsupport_BlockCyclicDistArray2dDouble_get(ab, i, j, &ex);
          double ab_ik = hplsupport_BlockCyclicDistArray2dDouble_get(ab, i, k, &ex);
          double ab_kj = hplsupport_BlockCyclicDistArray2dDouble_get(ab, k, j, &ex);
          double newVal = ab_ij - (ab_ik * ab_kj);
          hplsupport_BlockCyclicDistArray2dDouble_set(ab, newVal, i, j, &ex);
        }
      }
      if (DEBUG) printf(" k=%d loop ends \n", k);
    }
    for (int i = abStart1; i <= abEnd1; i++) {
      int32_t p1 = hplsupport_SimpleArray1dInt_get(piv, i, &ex);
      if (DEBUG) printf(" piv[%d] = %d \n", i, p1);
    }
    if (DEBUG) printf("impl_hpcc_HighPerformanceLinpack_panelSolveCompute(%p, %p) returns \n", ab, piv);
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack.panelSolveCompute) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

