/*
 * File:          hpcc_HighPerformanceLinpack_Skel.c
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "hpcc_HighPerformanceLinpack_IOR.h"
#include "hpcc_HighPerformanceLinpack.h"
#include <stddef.h>

#ifdef WITH_RMI
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_h
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#endif
#ifndef included_hplsupport_SimpleArray1dInt_h
#include "hplsupport_SimpleArray1dInt.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#endif /* WITH_RMI */
extern
void
impl_hpcc_HighPerformanceLinpack__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_HighPerformanceLinpack__ctor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_HighPerformanceLinpack__ctor2(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_HighPerformanceLinpack__dtor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex);

extern
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
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct hplsupport_SimpleArray1dInt__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_hplsupport_SimpleArray1dInt(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct hplsupport_SimpleArray1dInt__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_hplsupport_SimpleArray1dInt(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_hpcc_HighPerformanceLinpack_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef __cplusplus
extern "C" {
#endif

void
hpcc_HighPerformanceLinpack__set_epv(struct hpcc_HighPerformanceLinpack__epv 
  *epv,
  struct hpcc_HighPerformanceLinpack__pre_epv *pre_epv, 
  struct hpcc_HighPerformanceLinpack__post_epv *post_epv
)
{
  epv->f__ctor = impl_hpcc_HighPerformanceLinpack__ctor;
  epv->f__ctor2 = impl_hpcc_HighPerformanceLinpack__ctor2;
  epv->f__dtor = impl_hpcc_HighPerformanceLinpack__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
hpcc_HighPerformanceLinpack__set_sepv(struct hpcc_HighPerformanceLinpack__sepv 
  *sepv,
  struct hpcc_HighPerformanceLinpack__pre_sepv *pre_sepv, 
  struct hpcc_HighPerformanceLinpack__post_sepv *post_sepv)
{
  pre_sepv->f_panelSolveCompute_pre = NULL;
  sepv->f_panelSolveCompute = 
    impl_hpcc_HighPerformanceLinpack_panelSolveCompute;
  post_sepv->f_panelSolveCompute_post = NULL;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void hpcc_HighPerformanceLinpack__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_hpcc_HighPerformanceLinpack__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_hplsupport_BlockCyclicDistArray2dDouble
  (const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return hplsupport_BlockCyclicDistArray2dDouble__connectI(url, ar, _ex);
}

struct hplsupport_SimpleArray1dInt__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_hplsupport_SimpleArray1dInt(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return hplsupport_SimpleArray1dInt__connectI(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_hpcc_HighPerformanceLinpack_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct hpcc_HighPerformanceLinpack__data*
hpcc_HighPerformanceLinpack__get_data(hpcc_HighPerformanceLinpack self)
{
  return (struct hpcc_HighPerformanceLinpack__data*)(self ? self->d_data : 
    NULL);
}

void hpcc_HighPerformanceLinpack__set_data(
  hpcc_HighPerformanceLinpack self,
  struct hpcc_HighPerformanceLinpack__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
