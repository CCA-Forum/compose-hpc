/*
 * File:          hpcc_ParallelTranspose_Skel.c
 * Symbol:        hpcc.ParallelTranspose-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side glue code for hpcc.ParallelTranspose
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "hpcc_ParallelTranspose_IOR.h"
#include "hpcc_ParallelTranspose.h"
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
impl_hpcc_ParallelTranspose__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_ParallelTranspose__ctor(
  /* in */ hpcc_ParallelTranspose self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_ParallelTranspose__ctor2(
  /* in */ hpcc_ParallelTranspose self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_ParallelTranspose__dtor(
  /* in */ hpcc_ParallelTranspose self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hpcc_ParallelTranspose_ptransCompute(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble a,
  /* in */ hplsupport_BlockCyclicDistArray2dDouble c,
  /* in */ double beta,
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hpcc_ParallelTranspose_fconnect_hplsupport_BlockCyclicDistArray2dDouble(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_hpcc_ParallelTranspose_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef WITH_RMI
extern struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  impl_hpcc_ParallelTranspose_fconnect_hplsupport_BlockCyclicDistArray2dDouble(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_hpcc_ParallelTranspose_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
#ifdef __cplusplus
extern "C" {
#endif

void
hpcc_ParallelTranspose__set_epv(struct hpcc_ParallelTranspose__epv *epv,
  struct hpcc_ParallelTranspose__pre_epv *pre_epv, 
  struct hpcc_ParallelTranspose__post_epv *post_epv
)
{
  epv->f__ctor = impl_hpcc_ParallelTranspose__ctor;
  epv->f__ctor2 = impl_hpcc_ParallelTranspose__ctor2;
  epv->f__dtor = impl_hpcc_ParallelTranspose__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
hpcc_ParallelTranspose__set_sepv(struct hpcc_ParallelTranspose__sepv *sepv,
  struct hpcc_ParallelTranspose__pre_sepv *pre_sepv, 
  struct hpcc_ParallelTranspose__post_sepv *post_sepv)
{
  pre_sepv->f_ptransCompute_pre = NULL;
  sepv->f_ptransCompute = impl_hpcc_ParallelTranspose_ptransCompute;
  post_sepv->f_ptransCompute_post = NULL;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void hpcc_ParallelTranspose__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_hpcc_ParallelTranspose__load(&_throwaway_exception);
}
#ifdef WITH_RMI
struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  skel_hpcc_ParallelTranspose_fconnect_hplsupport_BlockCyclicDistArray2dDouble(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return hplsupport_BlockCyclicDistArray2dDouble__connectI(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_hpcc_ParallelTranspose_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
struct hpcc_ParallelTranspose__data*
hpcc_ParallelTranspose__get_data(hpcc_ParallelTranspose self)
{
  return (struct hpcc_ParallelTranspose__data*)(self ? self->d_data : NULL);
}

void hpcc_ParallelTranspose__set_data(
  hpcc_ParallelTranspose self,
  struct hpcc_ParallelTranspose__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
