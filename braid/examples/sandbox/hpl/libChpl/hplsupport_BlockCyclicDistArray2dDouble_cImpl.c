/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_Impl.c
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hplsupport.BlockCyclicDistArray2dDouble" (version 0.1)
 */

#include "hplsupport_BlockCyclicDistArray2dDouble_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._includes) */
#include <stdlib.h>
#include <string.h>
/* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble__ctor(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._ctor) */

	struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = (struct hplsupport_BlockCyclicDistArray2dDouble__data*)malloc(sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__data));
	if (dptr) {
	  memset(dptr, 0, sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__data));
	  // initialize elements of dptr here
	  hplsupport_BlockCyclicDistArray2dDouble__set_data(self, dptr);
	} else {
	  sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
	  SIDL_CHECK(*_ex);
	  sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
	  sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hplsupport.BlockCyclicDistArray2dDouble._ctor", _ex);
	  SIDL_CHECK(*_ex);
	  *_ex = (sidl_BaseInterface)ex;
	}
	EXIT:;

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble__ctor2(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble__dtor(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._dtor) */

    struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
    if (dptr) {
      // free contained in dtor before next line
      free(dptr);
      hplsupport_BlockCyclicDistArray2dDouble__set_data(self, NULL);
    }

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._dtor) */
  }
}

/*
 * Method:  initData[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_initData"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble_initData(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ void* data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.initData) */

	struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
	dptr->chpl_data = (BlockCyclicDistArray2dDoubleChpl) data;

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.initData) */
  }
}

/*
 * Method:  getFromArray[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray"

#ifdef __cplusplus
extern "C"
#endif
extern
double
impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(
		BlockCyclicDistArray2dDoubleChpl wrappedArray,
		/* in */ int32_t idx1,
		/* in */ int32_t idx2);

#ifdef __cplusplus
extern "C"
#endif
double
impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.getFromArray) */

	struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
	return impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(dptr->chpl_data, idx1, idx2);

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.getFromArray) */
  }
}

/*
 * Method:  setIntoArray[]
 */

#ifdef __cplusplus
extern "C"
#endif
extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
		BlockCyclicDistArray2dDoubleChpl wrappedArray,
		/* in */ double newVal,
		/* in */ int32_t idx1,
		/* in */ int32_t idx2);

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.setIntoArray) */

	struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
	impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(dptr->chpl_data, newVal, idx1, idx2);

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.setIntoArray) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

