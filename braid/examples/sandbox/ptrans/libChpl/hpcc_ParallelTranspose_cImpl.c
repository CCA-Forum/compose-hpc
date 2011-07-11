/*
 * File:          hpcc_ParallelTranspose_Impl.c
 * Symbol:        hpcc.ParallelTranspose-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hpcc.ParallelTranspose
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hpcc.ParallelTranspose" (version 0.1)
 */

#include "hpcc_ParallelTranspose_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._includes) */
/* insert code here (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_ParallelTranspose__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_ParallelTranspose__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_ParallelTranspose__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_ParallelTranspose__ctor(
  /* in */ hpcc_ParallelTranspose self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._ctor) */
    /*
     * // boilerplate constructor
     * struct hpcc_ParallelTranspose__data *dptr = (struct hpcc_ParallelTranspose__data*)malloc(sizeof(struct hpcc_ParallelTranspose__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct hpcc_ParallelTranspose__data));
     *   // initialize elements of dptr here
     * hpcc_ParallelTranspose__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hpcc.ParallelTranspose._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

    /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_ParallelTranspose__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_ParallelTranspose__ctor2(
  /* in */ hpcc_ParallelTranspose self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_ParallelTranspose__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_ParallelTranspose__dtor(
  /* in */ hpcc_ParallelTranspose self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._dtor) */
    /*
     * // boilerplate destructor
     * struct hpcc_ParallelTranspose__data *dptr = hpcc_ParallelTranspose__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   hpcc_ParallelTranspose__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._dtor) */
  }
}

/*
 * Method:  ptransCompute[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_ParallelTranspose_ptransCompute"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_ParallelTranspose_ptransCompute(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble a,
  /* in */ hplsupport_BlockCyclicDistArray2dDouble c,
  /* in */ double beta,
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose.ptransCompute) */
    double a_ji = hplsupport_BlockCyclicDistArray2dDouble_get(a, j, i, _ex);
    double c_ij = hplsupport_BlockCyclicDistArray2dDouble_get(c, i, j, _ex);

    double new_val = beta * c_ij  +  a_ji;

    hplsupport_BlockCyclicDistArray2dDouble_set(c, new_val, i, j, _ex);
    /* DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose.ptransCompute) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

