/*
 * File:          hplsupport_SimpleArray1dInt_Impl.h
 * Symbol:        hplsupport.SimpleArray1dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.SimpleArray1dInt
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_hplsupport_SimpleArray1dInt_Impl_h
#define included_hplsupport_SimpleArray1dInt_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
/* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._hincludes) */

/**
 * START: Chapel implementation specific declarations
 */
/**
 * END: Chapel implementation specific declarations
 */

/* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._hincludes) */

/*
 * Private data for class hplsupport.SimpleArray1dInt
 */

struct hplsupport_SimpleArray1dInt__data {
  /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._data) */
  /* insert code here (private data members) */
  int32_t* chpl_data;
  /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct hplsupport_SimpleArray1dInt__data*
hplsupport_SimpleArray1dInt__get_data(
  hplsupport_SimpleArray1dInt);

extern void
hplsupport_SimpleArray1dInt__set_data(
  hplsupport_SimpleArray1dInt,
  struct hplsupport_SimpleArray1dInt__data*);

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

/*
 * User-defined object methods
 */

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

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
