/*
 * File:          hpcc_HighPerformanceLinpack_Impl.h
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_hpcc_HighPerformanceLinpack_Impl_h
#define included_hpcc_HighPerformanceLinpack_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_hpcc_HighPerformanceLinpack_h
#include "hpcc_HighPerformanceLinpack.h"
#endif
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_h
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#endif
#ifndef included_hplsupport_SimpleArray1dInt_h
#include "hplsupport_SimpleArray1dInt.h"
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
/* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._hincludes) */

/*
 * Private data for class hpcc.HighPerformanceLinpack
 */

struct hpcc_HighPerformanceLinpack__data {
  /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._data) */
  /* insert code here (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct hpcc_HighPerformanceLinpack__data*
hpcc_HighPerformanceLinpack__get_data(
  hpcc_HighPerformanceLinpack);

extern void
hpcc_HighPerformanceLinpack__set_data(
  hpcc_HighPerformanceLinpack,
  struct hpcc_HighPerformanceLinpack__data*);

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

/*
 * User-defined object methods
 */

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

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
