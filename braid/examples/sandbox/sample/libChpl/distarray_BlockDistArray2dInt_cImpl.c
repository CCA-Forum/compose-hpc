/*
 * File:          distarray_BlockDistArray2dInt_Impl.c
 * Symbol:        distarray.BlockDistArray2dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for distarray.BlockDistArray2dInt
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "distarray.BlockDistArray2dInt" (version 0.1)
 */

#include "distarray_BlockDistArray2dInt_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._includes) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._load) */
	printf("  impl_distarray_BlockDistArray2dInt__load()\n");
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt__ctor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._ctor) */
	printf("  impl_distarray_BlockDistArray2dInt__ctor()\n");
	// boilerplate constructor
	struct distarray_BlockDistArray2dInt__data *dptr = (struct distarray_BlockDistArray2dInt__data*)malloc(sizeof(struct distarray_BlockDistArray2dInt__data));
	if (dptr) {
	  memset(dptr, 0, sizeof(struct distarray_BlockDistArray2dInt__data));
	  // initialize elements of dptr here
	  distarray_BlockDistArray2dInt__set_data(self, dptr);
	} else {
	  sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
	  SIDL_CHECK(*_ex);
	  sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
	  sidl_MemAllocException_add(ex, __FILE__, __LINE__, "distarray.BlockDistArray2dInt._ctor", _ex);
	  SIDL_CHECK(*_ex);
	  *_ex = (sidl_BaseInterface)ex;
	}
	EXIT:;
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt__ctor2(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._ctor2) */
	printf("  impl_distarray_BlockDistArray2dInt__ctor2()\n");
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt__dtor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._dtor) */
	printf("  impl_distarray_BlockDistArray2dInt__dtor()\n");
    // boilerplate destructor
    struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
    if (dptr) {
      // TODO Need to callback chapel to free the data?
      if (dptr->lower) free(dptr->lower);
      if (dptr->higher) free(dptr->higher);
      if (dptr->blocks) free(dptr->blocks);
      // free contained in dtor before next line
      free(dptr);
      distarray_BlockDistArray2dInt__set_data(self, NULL);
    }
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._dtor) */
  }
}

/*
 * Method:  matrixMultipleCannon[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_matrixMultipleCannon"


#ifdef __cplusplus
extern "C"
#endif
extern
void
impl_distarray_BlockDistArray2dInt_multiply_cannon_chpl(
		BlockDistArray2dIntChpl A,
		BlockDistArray2dIntChpl B,
		BlockDistArray2dIntChpl C,
		/* in */ int32_t lo1,
		/* in */ int32_t hi1,
		/* in */ int32_t lo2,
		/* in */ int32_t hi2,
		/* in */ int32_t blk1,
		/* in */ int32_t blk2);

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt_matrixMultipleCannon(
  /* inout */ distarray_BlockDistArray2dInt* A,
  /* inout */ distarray_BlockDistArray2dInt* B,
  /* inout */ distarray_BlockDistArray2dInt* C,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.matrixMultipleCannon) */
	printf("  impl_distarray_BlockDistArray2dInt_matrixMultipleCannon()\n");
	struct distarray_BlockDistArray2dInt__data *aDptr = distarray_BlockDistArray2dInt__get_data(*A);
	struct distarray_BlockDistArray2dInt__data *bDptr = distarray_BlockDistArray2dInt__get_data(*B);
	struct distarray_BlockDistArray2dInt__data *cDptr = distarray_BlockDistArray2dInt__get_data(*C);
	printf("  impl_distarray_BlockDistArray2dInt_matrixMultipleCannon(): checkpoint-1\n");
	printf("  A[1, 1] = %d", impl_distarray_BlockDistArray2dInt_getFromArray(*A, 1, 1, _ex));
	printf("  B[1, 1] = %d", impl_distarray_BlockDistArray2dInt_getFromArray(*B, 1, 1, _ex));
	printf("  C[1, 1] = %d", impl_distarray_BlockDistArray2dInt_getFromArray(*C, 1, 1, _ex));
    // TODO validate values for all 3 matrices. For now assume they share the values
	int32_t lo1 = cDptr->lower[0];
	int32_t hi1 = cDptr->higher[0];
	int32_t lo2 = cDptr->lower[1];
	int32_t hi2 = cDptr->higher[1];
	int32_t blk1 = cDptr->blocks[0];
	int32_t blk2 = cDptr->blocks[1];
	printf("  impl_distarray_BlockDistArray2dInt_matrixMultipleCannon(): checkpoint-2\n");
	impl_distarray_BlockDistArray2dInt_multiply_cannon_chpl(
			aDptr->chpl_data, bDptr->chpl_data, cDptr->chpl_data,
			lo1, hi1, lo2, hi2, blk1, blk2);
	return;
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.matrixMultipleCannon) */
  }
}

/*
 * Method:  initArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_initArray"

#ifdef __cplusplus
extern "C"
#endif
extern
BlockDistArray2dIntChpl impl_distarray_BlockDistArray2dInt_initArray_chpl(
		/* in */ int32_t lo1,
		/* in */ int32_t hi1,
		/* in */ int32_t lo2,
		/* in */ int32_t hi2,
		/* in */ int32_t blk1,
		/* in */ int32_t blk2);

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt_initArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t lo1,
  /* in */ int32_t hi1,
  /* in */ int32_t lo2,
  /* in */ int32_t hi2,
  /* in */ int32_t blk1,
  /* in */ int32_t blk2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.initArray) */
	printf("  impl_distarray_BlockDistArray2dInt_initArray()\n");
	BlockDistArray2dIntChpl distArray = impl_distarray_BlockDistArray2dInt_initArray_chpl(lo1, hi1, lo2, hi2, blk1, blk2);
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	dptr->chpl_data = distArray;
	// TODO throw error when blk values do not even divide the array in each dimension
	// TODO Store inside metadata struct instead
	int32_t dimValue = 2;
	dptr->dimension = dimValue;
	dptr->lower = (int32_t*) malloc(sizeof(int32_t) * dimValue);
	dptr->lower[0] = lo1;
	dptr->lower[1] = lo2;
	dptr->higher = (int32_t*) malloc(sizeof(int32_t) * dimValue);
	dptr->higher[0] = hi1;
	dptr->higher[1] = hi2;
	dptr->blocks = (int32_t*) malloc(sizeof(int32_t) * dimValue);
	dptr->blocks[0] = (hi1 - lo1 + 1) / blk1;
	dptr->blocks[1] = (hi2 - lo2 + 1) / blk2;
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.initArray) */
  }
}

/*
 * Method:  getDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_getDimension"


#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_distarray_BlockDistArray2dInt_getDimension(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.getDimension) */
	printf("  impl_distarray_BlockDistArray2dInt_getDimension()\n");
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	return dptr->dimension;
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.getDimension) */
  }
}

/*
 * Method:  getLower[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_getLower"


#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_distarray_BlockDistArray2dInt_getLower(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.getLower) */
	printf("  impl_distarray_BlockDistArray2dInt_getLower()\n");
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	return dptr->lower[dim];
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.getLower) */
  }
}

/*
 * Method:  getHigher[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_getHigher"


#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_distarray_BlockDistArray2dInt_getHigher(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.getHigher) */
	printf("  impl_distarray_BlockDistArray2dInt_getHigher()\n");
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	return dptr->higher[dim];
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.getHigher) */
  }
}

/*
 * Method:  getFromArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_getFromArray"

#ifdef __cplusplus
extern "C"
#endif
extern
int32_t
impl_distarray_BlockDistArray2dInt_getFromArray_chpl(
		BlockDistArray2dIntChpl distArray,
		/* in */ int32_t idx1,
		/* in */ int32_t idx2);

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_distarray_BlockDistArray2dInt_getFromArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.getFromArray) */
	printf("  impl_distarray_BlockDistArray2dInt_getFromArray()\n");
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	return impl_distarray_BlockDistArray2dInt_getFromArray_chpl(dptr->chpl_data, idx1, idx2);
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.getFromArray) */
  }
}

/*
 * Method:  setIntoArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_distarray_BlockDistArray2dInt_setIntoArray"

#ifdef __cplusplus
extern "C"
#endif
extern
int32_t
impl_distarray_BlockDistArray2dInt_setIntoArray_chpl(
		BlockDistArray2dIntChpl distArray,
		/* in */ int32_t newVal,
		/* in */ int32_t idx1,
		/* in */ int32_t idx2);

#ifdef __cplusplus
extern "C"
#endif
void
impl_distarray_BlockDistArray2dInt_setIntoArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.setIntoArray) */
	printf("  impl_distarray_BlockDistArray2dInt_setIntoArray()\n");
	struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
	impl_distarray_BlockDistArray2dInt_setIntoArray_chpl(dptr->chpl_data, newVal, idx1, idx2);
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.setIntoArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

