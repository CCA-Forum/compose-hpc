/*
 * File:          hplsupport_SimpleArray1dInt_Impl.c
 * Symbol:        hplsupport.SimpleArray1dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.SimpleArray1dInt
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hplsupport.SimpleArray1dInt" (version 0.1)
 */

#include "hplsupport_SimpleArray1dInt_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._includes) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt__ctor(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._ctor) */

    // boilerplate constructor
    struct hplsupport_SimpleArray1dInt__data *dptr = (struct hplsupport_SimpleArray1dInt__data*)malloc(sizeof(struct hplsupport_SimpleArray1dInt__data));
    if (dptr) {
      memset(dptr, 0, sizeof(struct hplsupport_SimpleArray1dInt__data));
      // initialize elements of dptr here
      hplsupport_SimpleArray1dInt__set_data(self, dptr);
    } else {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hplsupport.SimpleArray1dInt._ctor", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (sidl_BaseInterface)ex;
    }
    EXIT:;

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt__ctor2(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt__dtor(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt._dtor) */

	// boilerplate destructor
    struct hplsupport_SimpleArray1dInt__data *dptr = hplsupport_SimpleArray1dInt__get_data(self);
    if (dptr) {
      // free contained in dtor before next line
      free(dptr);
      hplsupport_SimpleArray1dInt__set_data(self, NULL);
    }

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt._dtor) */
  }
}

/*
 * Method:  initData[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt_initData"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt_initData(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ void* data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt.initData) */

	struct hplsupport_SimpleArray1dInt__data *dptr = hplsupport_SimpleArray1dInt__get_data(self);
	SimpleArray1dIntChpl* chplDataPtr = (SimpleArray1dIntChpl*) data;
	dptr->chpl_data = (*chplDataPtr);

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt.initData) */
  }
}

/*
 * Method:  getFromArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt_getFromArray"

#ifdef __cplusplus
extern "C"
#endif
extern
int32_t
impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(
		SimpleArray1dIntChpl wrappedArray,
		/* in */ int32_t idx1);

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_hplsupport_SimpleArray1dInt_getFromArray(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt.getFromArray) */

	// printf("impl_hplsupport_SimpleArray1dInt_getFromArray(%d)\n", idx1);
	struct hplsupport_SimpleArray1dInt__data *dptr = hplsupport_SimpleArray1dInt__get_data(self);
	return impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(dptr->chpl_data, idx1);

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt.getFromArray) */
  }
}

/*
 * Method:  setIntoArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_SimpleArray1dInt_setIntoArray"

#ifdef __cplusplus
extern "C"
#endif
extern
void
impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(
		SimpleArray1dIntChpl wrappedArray,
		/* in */ int32_t newVal,
		/* in */ int32_t idx1);

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_SimpleArray1dInt_setIntoArray(
  /* in */ hplsupport_SimpleArray1dInt self,
  /* in */ int32_t newVal,
  /* in */ int32_t idx1,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.SimpleArray1dInt.setIntoArray) */
	// printf("impl_hplsupport_SimpleArray1dInt_setIntoArray(%d) = %d\n", idx1, newVal);
	struct hplsupport_SimpleArray1dInt__data *dptr = hplsupport_SimpleArray1dInt__get_data(self);
	impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(dptr->chpl_data, newVal, idx1);

    /* DO-NOT-DELETE splicer.end(hplsupport.SimpleArray1dInt.setIntoArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

