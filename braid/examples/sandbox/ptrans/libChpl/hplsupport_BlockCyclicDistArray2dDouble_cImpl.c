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
#include <stdio.h>
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


#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_ptransHelper"

/**
 * FIXME: The impl method signature does not have pointers unlike the function pointer in the epv struct.
 * There is a corresponding FIXME in the IOR.h. It needs to be fixed one of these two places.
 */
#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble_ptransHelper(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble* a,
  /* inout */ hplsupport_BlockCyclicDistArray2dDouble** c,
  /* in */ double beta,
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.ptransHelper) */
    double a_ji = hplsupport_BlockCyclicDistArray2dDouble_get(*a, j, i, _ex);
    double c_ij = hplsupport_BlockCyclicDistArray2dDouble_get(*(*c), i, j, _ex);

    double new_val = beta * c_ij  +  a_ji;

    hplsupport_BlockCyclicDistArray2dDouble_set(*(*c), new_val, i, j, _ex);
    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.ptransHelper) */
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
    int32_t* chplDataPtr = (int32_t*) data;
    dptr->chpl_data = *chplDataPtr;
    // printf("impl_hplsupport_BlockCyclicDistArray2dDouble_initData(): chplDataPtr = %p, chpl_data = %d \n", chplDataPtr, dptr->chpl_data);

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.initData) */
  }
}

/*
 * Method:  get[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_get"

#ifdef __cplusplus
extern "C"
#endif
extern
double
impl_hplsupport_BlockCyclicDistArray2dDouble_get_chpl(
        int32_t chplArray,
        /* in */ int32_t idx1,
        /* in */ int32_t idx2);

#ifdef __cplusplus
extern "C"
#endif
double
impl_hplsupport_BlockCyclicDistArray2dDouble_get(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.get) */

    // printf("impl_hplsupport_BlockCyclicDistArray2dDouble_get(%d, %d)\n", idx1, idx2);
    struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
    // printf("impl_hplsupport_BlockCyclicDistArray2dDouble_get(): chpl_data = %p \n", dptr->chpl_data);
    return impl_hplsupport_BlockCyclicDistArray2dDouble_get_chpl(dptr->chpl_data, idx1, idx2);

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.get) */
  }
}

/*
 * Method:  set[]
 */

#ifdef __cplusplus
extern "C"
#endif
extern
void
impl_hplsupport_BlockCyclicDistArray2dDouble_set_chpl(
        int32_t chplArray,
        /* in */ double newVal,
        /* in */ int32_t idx1,
        /* in */ int32_t idx2);

#undef __FUNC__
#define __FUNC__ "impl_hplsupport_BlockCyclicDistArray2dDouble_set"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hplsupport_BlockCyclicDistArray2dDouble_set(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.set) */

    // printf("impl_hplsupport_BlockCyclicDistArray2dDouble_set(%d, %d) = %f\n", idx1, idx2, newVal);
    struct hplsupport_BlockCyclicDistArray2dDouble__data *dptr = hplsupport_BlockCyclicDistArray2dDouble__get_data(self);
    // printf("impl_hplsupport_BlockCyclicDistArray2dDouble_set(): chpl_data = %d \n", dptr->chpl_data);
    impl_hplsupport_BlockCyclicDistArray2dDouble_set_chpl(dptr->chpl_data, newVal, idx1, idx2);

    /* DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.set) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

