/*
 * File:          hpcc_ParallelTranspose_Impl.h
 * Symbol:        hpcc.ParallelTranspose-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hpcc.ParallelTranspose
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_hpcc_ParallelTranspose_Impl_h
#define included_hpcc_ParallelTranspose_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_hpcc_ParallelTranspose_h
#include "hpcc_ParallelTranspose.h"
#endif
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_h
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
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
/* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._hincludes) */

/*
 * Private data for class hpcc.ParallelTranspose
 */

struct hpcc_ParallelTranspose__data {
  /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._data) */
  /* insert code here (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct hpcc_ParallelTranspose__data*
hpcc_ParallelTranspose__get_data(
  hpcc_ParallelTranspose);

extern void
hpcc_ParallelTranspose__set_data(
  hpcc_ParallelTranspose,
  struct hpcc_ParallelTranspose__data*);

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

/*
 * User-defined object methods
 */

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

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
