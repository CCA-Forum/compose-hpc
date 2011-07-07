/*
 * File:          hplsupport_SimpleArray1dInt_Skel.c
 * Symbol:        hplsupport.SimpleArray1dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for hplsupport.SimpleArray1dInt
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "hplsupport_SimpleArray1dInt_IOR.h"
#include "hplsupport_SimpleArray1dInt.h"
#include <stddef.h>

#ifdef WITH_RMI
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#endif /* WITH_RMI */
extern
void
impl_hplsupport_SimpleArray1dInt__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dInt__ctor(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dInt__ctor2(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dInt__dtor(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_SimpleArray1dInt_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
void
impl_hplsupport_SimpleArray1dInt_initData(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ void* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_hplsupport_SimpleArray1dInt_getFromArray(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dInt_setIntoArray(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ int32_t newVal,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_SimpleArray1dInt_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef __cplusplus
extern "C" {
#endif

void
hplsupport_SimpleArray1dInt__set_epv(struct hplsupport_SimpleArray1dInt__epv 
  *epv,
  struct hplsupport_SimpleArray1dInt__pre_epv *pre_epv, 
  struct hplsupport_SimpleArray1dInt__post_epv *post_epv
)
{
  epv->f__ctor = impl_hplsupport_SimpleArray1dInt__ctor;
  epv->f__ctor2 = impl_hplsupport_SimpleArray1dInt__ctor2;
  epv->f__dtor = impl_hplsupport_SimpleArray1dInt__dtor;
  pre_epv->f_initData_pre = NULL;
  epv->f_initData = impl_hplsupport_SimpleArray1dInt_initData;
  post_epv->f_initData_post = NULL;
  pre_epv->f_getFromArray_pre = NULL;
  epv->f_getFromArray = impl_hplsupport_SimpleArray1dInt_getFromArray;
  post_epv->f_getFromArray_post = NULL;
  pre_epv->f_setIntoArray_pre = NULL;
  epv->f_setIntoArray = impl_hplsupport_SimpleArray1dInt_setIntoArray;
  post_epv->f_setIntoArray_post = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void hplsupport_SimpleArray1dInt__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_hplsupport_SimpleArray1dInt__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct sidl_BaseInterface__object* 
  skel_hplsupport_SimpleArray1dInt_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct hplsupport_SimpleArray1dInt__data*
hplsupport_SimpleArray1dInt__get_data(hplsupport_SimpleArray1dInt self)
{
  return (struct hplsupport_SimpleArray1dInt__data*)(self ? self->d_data : 
    NULL);
}

void hplsupport_SimpleArray1dInt__set_data(
  hplsupport_SimpleArray1dInt self,
  struct hplsupport_SimpleArray1dInt__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
