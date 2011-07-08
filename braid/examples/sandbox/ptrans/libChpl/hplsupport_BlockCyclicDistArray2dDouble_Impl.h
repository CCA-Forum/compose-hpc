/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_Impl.h
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_Impl_h
#define included_hplsupport_BlockCyclicDistArray2dDouble_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
/* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._hincludes) */

/* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._hincludes) */

/*
 * Private data for class hplsupport.BlockCyclicDistArray2dDouble
 */

struct hplsupport_BlockCyclicDistArray2dDouble__data {
  /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._data) */
  /* insert code here (private data members) */
  int32_t chpl_data;
  /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct hplsupport_BlockCyclicDistArray2dDouble__data*
hplsupport_BlockCyclicDistArray2dDouble__get_data(
  hplsupport_BlockCyclicDistArray2dDouble);

extern void
hplsupport_BlockCyclicDistArray2dDouble__set_data(
  hplsupport_BlockCyclicDistArray2dDouble,
  struct hplsupport_BlockCyclicDistArray2dDouble__data*);

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

/*
 * User-defined object methods
 */

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

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
