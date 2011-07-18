/*
 * File:          blas_VectorUtils_Impl.h
 * Symbol:        blas.VectorUtils-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for blas.VectorUtils
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_blas_VectorUtils_Impl_h
#define included_blas_VectorUtils_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_blas_VectorUtils_h
#include "blas_VectorUtils.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
/* DO-NOT-DELETE splicer.begin(blas.VectorUtils._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(blas.VectorUtils._hincludes) */

/*
 * Private data for class blas.VectorUtils
 */

struct blas_VectorUtils__data {
  /* DO-NOT-DELETE splicer.begin(blas.VectorUtils._data) */
  /* insert code here (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(blas.VectorUtils._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct blas_VectorUtils__data*
blas_VectorUtils__get_data(
  blas_VectorUtils);

extern void
blas_VectorUtils__set_data(
  blas_VectorUtils,
  struct blas_VectorUtils__data*);

extern
void
impl_blas_VectorUtils__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_blas_VectorUtils__ctor(
  /* in */ blas_VectorUtils self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_blas_VectorUtils__ctor2(
  /* in */ blas_VectorUtils self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_blas_VectorUtils__dtor(
  /* in */ blas_VectorUtils self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
void
impl_blas_VectorUtils_helper_daxpy(
  /* in */ int32_t n,
  /* in */ double alpha,
  /* in array<double> */ struct sidl_double__array* X,
  /* inout array<double> */ struct sidl_double__array** Y,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_blas_VectorUtils_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_blas_VectorUtils_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
