/*
 * File:          hplsupport_SimpleArray1dDouble_Impl.h
 * Symbol:        hplsupport.SimpleArray1dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.SimpleArray1dDouble
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_hplsupport_SimpleArray1dDouble_Impl_h
#define included_hplsupport_SimpleArray1dDouble_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_hplsupport_SimpleArray1dDouble_h
#include "hplsupport_SimpleArray1dDouble.h"
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
/* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._hincludes) */
/* insert code here (include files) */
/* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._hincludes) */

/*
 * Private data for class hplsupport.SimpleArray1dDouble
 */

struct hplsupport_SimpleArray1dDouble__data {
  /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._data) */
  /* insert code here (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct hplsupport_SimpleArray1dDouble__data*
hplsupport_SimpleArray1dDouble__get_data(
  hplsupport_SimpleArray1dDouble);

extern void
hplsupport_SimpleArray1dDouble__set_data(
  hplsupport_SimpleArray1dDouble,
  struct hplsupport_SimpleArray1dDouble__data*);

extern
void
impl_hplsupport_SimpleArray1dDouble__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dDouble__ctor(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dDouble__ctor2(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dDouble__dtor(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_SimpleArray1dDouble_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
double
impl_hplsupport_SimpleArray1dDouble_getFromArray(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_hplsupport_SimpleArray1dDouble_setIntoArray(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_hplsupport_SimpleArray1dDouble_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
