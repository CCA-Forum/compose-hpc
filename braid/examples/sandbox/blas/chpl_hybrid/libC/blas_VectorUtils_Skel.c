/*
 * File:          blas_VectorUtils_Skel.c
 * Symbol:        blas.VectorUtils-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for blas.VectorUtils
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "blas_VectorUtils_IOR.h"
#include "blas_VectorUtils.h"
#include <stddef.h>

#ifdef WITH_RMI
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#endif /* WITH_RMI */
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
#ifdef __cplusplus
extern "C" {
#endif

void
blas_VectorUtils__set_epv(struct blas_VectorUtils__epv *epv,
  struct blas_VectorUtils__pre_epv *pre_epv, 
  struct blas_VectorUtils__post_epv *post_epv
)
{
  epv->f__ctor = impl_blas_VectorUtils__ctor;
  epv->f__ctor2 = impl_blas_VectorUtils__ctor2;
  epv->f__dtor = impl_blas_VectorUtils__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
blas_VectorUtils__set_sepv(struct blas_VectorUtils__sepv *sepv,
  struct blas_VectorUtils__pre_sepv *pre_sepv, 
  struct blas_VectorUtils__post_sepv *post_sepv)
{
  pre_sepv->f_helper_daxpy_pre = NULL;
  sepv->f_helper_daxpy = impl_blas_VectorUtils_helper_daxpy;
  post_sepv->f_helper_daxpy_post = NULL;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void blas_VectorUtils__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_blas_VectorUtils__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct sidl_BaseInterface__object* 
  skel_blas_VectorUtils_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct blas_VectorUtils__data*
blas_VectorUtils__get_data(blas_VectorUtils self)
{
  return (struct blas_VectorUtils__data*)(self ? self->d_data : NULL);
}

void blas_VectorUtils__set_data(
  blas_VectorUtils self,
  struct blas_VectorUtils__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
