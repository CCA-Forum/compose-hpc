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
/* insert code here (includes and arbitrary code) */
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
    /* insert code here (static class initializer) */
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
    /*
     * // boilerplate constructor
     * struct distarray_BlockDistArray2dInt__data *dptr = (struct distarray_BlockDistArray2dInt__data*)malloc(sizeof(struct distarray_BlockDistArray2dInt__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct distarray_BlockDistArray2dInt__data));
     *   // initialize elements of dptr here
     * distarray_BlockDistArray2dInt__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "distarray.BlockDistArray2dInt._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

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
    /*
     * // boilerplate destructor
     * struct distarray_BlockDistArray2dInt__data *dptr = distarray_BlockDistArray2dInt__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   distarray_BlockDistArray2dInt__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._dtor) */
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
    /* insert code here (initArray) */
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
    /* insert code here (getDimension) */
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
    /* insert code here (getLower) */
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
    /* insert code here (getHigher) */
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
void
impl_distarray_BlockDistArray2dInt_getFromArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt.getFromArray) */
    /* insert code here (getFromArray) */
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
    /* insert code here (setIntoArray) */
    /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt.setIntoArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

