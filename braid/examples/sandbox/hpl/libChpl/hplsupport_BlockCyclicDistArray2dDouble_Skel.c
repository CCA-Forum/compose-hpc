/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_Skel.c
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "hplsupport_BlockCyclicDistArray2dDouble_IOR.h"
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#include <stddef.h>

#ifdef WITH_RMI
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#endif /* WITH_RMI */
extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble__ctor(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble__ctor2(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble__dtor(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
double
impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef __cplusplus
extern "C" {
#endif

void
hplsupport_BlockCyclicDistArray2dDouble__set_epv(struct 
  hplsupport_BlockCyclicDistArray2dDouble__epv *epv,
  struct hplsupport_BlockCyclicDistArray2dDouble__pre_epv *pre_epv, 
  struct hplsupport_BlockCyclicDistArray2dDouble__post_epv *post_epv
)
{
  epv->f__ctor = impl_hplsupport_BlockCyclicDistArray2dDouble__ctor;
  epv->f__ctor2 = impl_hplsupport_BlockCyclicDistArray2dDouble__ctor2;
  epv->f__dtor = impl_hplsupport_BlockCyclicDistArray2dDouble__dtor;
  pre_epv->f_getFromArray_pre = NULL;
  epv->f_getFromArray = 
    impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray;
  post_epv->f_getFromArray_post = NULL;
  pre_epv->f_setIntoArray_pre = NULL;
  epv->f_setIntoArray = 
    impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray;
  post_epv->f_setIntoArray_post = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void hplsupport_BlockCyclicDistArray2dDouble__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_hplsupport_BlockCyclicDistArray2dDouble__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct sidl_BaseInterface__object* 
  skel_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct hplsupport_BlockCyclicDistArray2dDouble__data*
hplsupport_BlockCyclicDistArray2dDouble__get_data(
  hplsupport_BlockCyclicDistArray2dDouble self)
{
  return (struct hplsupport_BlockCyclicDistArray2dDouble__data*)(self ? 
    self->d_data : NULL);
}

void hplsupport_BlockCyclicDistArray2dDouble__set_data(
  hplsupport_BlockCyclicDistArray2dDouble self,
  struct hplsupport_BlockCyclicDistArray2dDouble__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
