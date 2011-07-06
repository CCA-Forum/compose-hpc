/*
 * File:          hplsupport_SimpleArray1dDouble_Impl.c
 * Symbol:        hplsupport.SimpleArray1dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.SimpleArray1dDouble
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hplsupport.SimpleArray1dDouble" (version 0.1)
 */

#include "hplsupport_SimpleArray1dDouble_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._includes) */
/* insert code here (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dDouble__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dDouble__ctor(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._ctor) */
    /*
     * // boilerplate constructor
     * struct hplsupport_SimpleArray1dDouble__data *dptr = (struct hplsupport_SimpleArray1dDouble__data*)malloc(sizeof(struct hplsupport_SimpleArray1dDouble__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct hplsupport_SimpleArray1dDouble__data));
     *   // initialize elements of dptr here
     * hplsupport_SimpleArray1dDouble__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hplsupport.SimpleArray1dDouble._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dDouble__ctor2(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dDouble__dtor(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble._dtor) */
    /*
     * // boilerplate destructor
     * struct hplsupport_SimpleArray1dDouble__data *dptr = hplsupport_SimpleArray1dDouble__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   hplsupport_SimpleArray1dDouble__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble._dtor) */
  }
}

/*
 * Method:  getFromArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble_getFromArray"

#ifdef __cplusplus
extern "C"
#endif
double
impl_hplsupport_SimpleArray1dDouble_getFromArray(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble.getFromArray) */
    /* insert code here (getFromArray) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble.getFromArray) */
  }
}

/*
 * Method:  setIntoArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dDouble_setIntoArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dDouble_setIntoArray(
  /* in */ hplsupport_SimpleArray1dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dDouble.setIntoArray) */
    /* insert code here (setIntoArray) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dDouble.setIntoArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

