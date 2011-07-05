/*
 * File:          distarray_BlockDistArray2dInt_Skel.c
 * Symbol:        distarray.BlockDistArray2dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for distarray.BlockDistArray2dInt
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "distarray_BlockDistArray2dInt_IOR.h"
#include "distarray_BlockDistArray2dInt.h"
#include <stddef.h>

#ifdef WITH_RMI
#ifndef included_distarray_BlockDistArray2dInt_h
#include "distarray_BlockDistArray2dInt.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#endif /* WITH_RMI */
extern
void
impl_distarray_BlockDistArray2dInt__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__ctor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__ctor2(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__dtor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt_matrixMultipleCannon(
  /* inout */ distarray_BlockDistArray2dInt* A,
  /* inout */ distarray_BlockDistArray2dInt* B,
  /* inout */ distarray_BlockDistArray2dInt* C,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct distarray_BlockDistArray2dInt__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_distarray_BlockDistArray2dInt(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
void
impl_distarray_BlockDistArray2dInt_initArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t lo1,
  /* in */ int32_t hi1,
  /* in */ int32_t lo2,
  /* in */ int32_t hi2,
  /* in */ int32_t blk1,
  /* in */ int32_t blk2,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getDimension(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getLower(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getHigher(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getFromArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt_setIntoArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct distarray_BlockDistArray2dInt__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_distarray_BlockDistArray2dInt(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef __cplusplus
extern "C" {
#endif

void
distarray_BlockDistArray2dInt__set_epv(struct 
  distarray_BlockDistArray2dInt__epv *epv,
  struct distarray_BlockDistArray2dInt__pre_epv *pre_epv, 
  struct distarray_BlockDistArray2dInt__post_epv *post_epv
)
{
  epv->f__ctor = impl_distarray_BlockDistArray2dInt__ctor;
  epv->f__ctor2 = impl_distarray_BlockDistArray2dInt__ctor2;
  epv->f__dtor = impl_distarray_BlockDistArray2dInt__dtor;
  pre_epv->f_initArray_pre = NULL;
  epv->f_initArray = impl_distarray_BlockDistArray2dInt_initArray;
  post_epv->f_initArray_post = NULL;
  pre_epv->f_getDimension_pre = NULL;
  epv->f_getDimension = impl_distarray_BlockDistArray2dInt_getDimension;
  post_epv->f_getDimension_post = NULL;
  pre_epv->f_getLower_pre = NULL;
  epv->f_getLower = impl_distarray_BlockDistArray2dInt_getLower;
  post_epv->f_getLower_post = NULL;
  pre_epv->f_getHigher_pre = NULL;
  epv->f_getHigher = impl_distarray_BlockDistArray2dInt_getHigher;
  post_epv->f_getHigher_post = NULL;
  pre_epv->f_getFromArray_pre = NULL;
  epv->f_getFromArray = impl_distarray_BlockDistArray2dInt_getFromArray;
  post_epv->f_getFromArray_post = NULL;
  pre_epv->f_setIntoArray_pre = NULL;
  epv->f_setIntoArray = impl_distarray_BlockDistArray2dInt_setIntoArray;
  post_epv->f_setIntoArray_post = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
distarray_BlockDistArray2dInt__set_sepv(struct 
  distarray_BlockDistArray2dInt__sepv *sepv,
  struct distarray_BlockDistArray2dInt__pre_sepv *pre_sepv, 
  struct distarray_BlockDistArray2dInt__post_sepv *post_sepv)
{
  pre_sepv->f_matrixMultipleCannon_pre = NULL;
  sepv->f_matrixMultipleCannon = 
    impl_distarray_BlockDistArray2dInt_matrixMultipleCannon;
  post_sepv->f_matrixMultipleCannon_post = NULL;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void distarray_BlockDistArray2dInt__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_distarray_BlockDistArray2dInt__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct distarray_BlockDistArray2dInt__object* 
  skel_distarray_BlockDistArray2dInt_fconnect_distarray_BlockDistArray2dInt(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return distarray_BlockDistArray2dInt__connectI(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct distarray_BlockDistArray2dInt__data*
distarray_BlockDistArray2dInt__get_data(distarray_BlockDistArray2dInt self)
{
  return (struct distarray_BlockDistArray2dInt__data*)(self ? self->d_data : 
    NULL);
}

void distarray_BlockDistArray2dInt__set_data(
  distarray_BlockDistArray2dInt self,
  struct distarray_BlockDistArray2dInt__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
