/*
 * File:          hpcc_HighPerformanceLinpack_Impl.c
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "hpcc.HighPerformanceLinpack" (version 0.1)
 */

#include "hpcc_HighPerformanceLinpack_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif

/* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._includes) */
/* insert code here (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._includes) */

#define SIDL_IOR_MAJOR_VERSION 2
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._load) */
    /* insert code here (static class initializer) */
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__ctor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._ctor) */
    /*
     * // boilerplate constructor
     * struct hpcc_HighPerformanceLinpack__data *dptr = (struct hpcc_HighPerformanceLinpack__data*)malloc(sizeof(struct hpcc_HighPerformanceLinpack__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct hpcc_HighPerformanceLinpack__data));
     *   // initialize elements of dptr here
     * hpcc_HighPerformanceLinpack__set_data(self, dptr);
     * } else {
     *   sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(_ex);
     *   SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
     *   sidl_MemAllocException_add(ex, __FILE__, __LINE__, "hpcc.HighPerformanceLinpack._ctor", _ex);
     *   SIDL_CHECK(*_ex);
     *   *_ex = (sidl_BaseInterface)ex;
     * }
     * EXIT:;
     */

    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__ctor2(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._ctor2) */
    /* insert code here (special constructor) */
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_hpcc_HighPerformanceLinpack__dtor(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack._dtor) */
    /*
     * // boilerplate destructor
     * struct hpcc_HighPerformanceLinpack__data *dptr = hpcc_HighPerformanceLinpack__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   hpcc_HighPerformanceLinpack__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack._dtor) */
  }
}

/*
 * Method:  panelSolveCompute[]
 */

#undef __FUNC__
#define __FUNC__ "impl_hpcc_HighPerformanceLinpack_panelSolveCompute"

#ifdef __cplusplus
extern "C"
#endif
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(hpcc.HighPerformanceLinpack.panelSolveCompute) */
    /* insert code here (panelSolveCompute) */
    /* DO-NOT-DELETE splicer.end(hpcc.HighPerformanceLinpack.panelSolveCompute) */
  }
}
/* Babel internal methods, Users should not edit below this line. */

/* DO-NOT-DELETE splicer.begin(_misc) */
/* insert code here (miscellaneous code) */
/* DO-NOT-DELETE splicer.end(_misc) */

