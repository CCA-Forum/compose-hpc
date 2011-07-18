/*
 * File:          blas_VectorUtils_Impl.c
 * Symbol:        blas.VectorUtils-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for blas.VectorUtils
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "blas.VectorUtils" (version 0.1)
 */

#include "blas_VectorUtils_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(blas.VectorUtils._includes) */
#include "cblas.h"
/* DO-NOT-DELETE splicer.end(blas.VectorUtils._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_blas_VectorUtils__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_blas_VectorUtils__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(blas.VectorUtils._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(blas.VectorUtils._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_blas_VectorUtils__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_blas_VectorUtils__ctor(
  /* in */ blas_VectorUtils self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(blas.VectorUtils._ctor) */
    /*
     * // boilerplate constructor
     * struct blas_VectorUtils__data *dptr = (struct blas_VectorUtils__data*)malloc(sizeof(struct blas_VectorUtils__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct blas_VectorUtils__data));
     *   // initialize elements of dptr here
     * blas_VectorUtils__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "blas.VectorUtils._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

    /* DO-NOT-DELETE splicer.end(blas.VectorUtils._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_blas_VectorUtils__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_blas_VectorUtils__ctor2(
  /* in */ blas_VectorUtils self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(blas.VectorUtils._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(blas.VectorUtils._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_blas_VectorUtils__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_blas_VectorUtils__dtor(
  /* in */ blas_VectorUtils self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(blas.VectorUtils._dtor) */
    /*
     * // boilerplate destructor
     * struct blas_VectorUtils__data *dptr = blas_VectorUtils__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   blas_VectorUtils__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(blas.VectorUtils._dtor) */
  }
}

/*
 * Method:  helper_daxpy[]
 */

#undef __FUNC__
#define __FUNC__ "impl_blas_VectorUtils_helper_daxpy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_blas_VectorUtils_helper_daxpy(
  /* in */ int32_t n,
  /* in */ double alpha,
  /* in array<double> */ struct sidl_double__array* X,
  /* inout array<double> */ struct sidl_double__array** Y,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(blas.VectorUtils.helper_daxpy) */
    double* xPtr = X->d_firstElement;
    double* yPtr = (*Y)->d_firstElement;
    cblas_daxpy(n, alpha, xPtr, 1, yPtr, 1);
    /* DO-NOT-DELETE splicer.end(blas.VectorUtils.helper_daxpy) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

