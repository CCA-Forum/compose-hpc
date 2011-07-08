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
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_h
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#endif
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

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_ptransHelper(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble a,
  /* inout */ hplsupport_BlockCyclicDistArray2dDouble* c,
  /* in */ double beta,
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hplsupport_BlockCyclicDistArray2dDouble_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_initData(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ void* data,
  /* out */ sidl_BaseInterface *_ex);

extern
double
impl_hplsupport_BlockCyclicDistArray2dDouble_get(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_set(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hplsupport_BlockCyclicDistArray2dDouble_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
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
  pre_epv->f_initData_pre = NULL;
  epv->f_initData = impl_hplsupport_BlockCyclicDistArray2dDouble_initData;
  post_epv->f_initData_post = NULL;
  pre_epv->f_get_pre = NULL;
  epv->f_get = 
    impl_hplsupport_BlockCyclicDistArray2dDouble_get;
  post_epv->f_get_post = NULL;
  pre_epv->f_set_pre = NULL;
  epv->f_set = 
    impl_hplsupport_BlockCyclicDistArray2dDouble_set;
  post_epv->f_set_post = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
hplsupport_BlockCyclicDistArray2dDouble__set_sepv(struct 
  hplsupport_BlockCyclicDistArray2dDouble__sepv *sepv,
  struct hplsupport_BlockCyclicDistArray2dDouble__pre_sepv *pre_sepv, 
  struct hplsupport_BlockCyclicDistArray2dDouble__post_sepv *post_sepv)
{
  pre_sepv->f_ptransHelper_pre = NULL;
  sepv->f_ptransHelper = 
    impl_hplsupport_BlockCyclicDistArray2dDouble_ptransHelper;
  post_sepv->f_ptransHelper_post = NULL;
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
struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  skel_hplsupport_BlockCyclicDistArray2dDouble_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return hplsupport_BlockCyclicDistArray2dDouble__connectI(url, ar, _ex);
}

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
