/*
 * File:          distarray_BlockDistArray2dInt_Impl.h
 * Symbol:        distarray.BlockDistArray2dInt-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Server-side implementation for distarray.BlockDistArray2dInt
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_distarray_BlockDistArray2dInt_Impl_h
#define included_distarray_BlockDistArray2dInt_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_distarray_BlockDistArray2dInt_h
#include "distarray_BlockDistArray2dInt.h"
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
/* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._hincludes) */

/**
 * START: Chapel implementation specific declarations
 */

// Make forward references of external structs
struct __DistArray_int32_t_2__array_BlockArr_int32_t_2_int32_t_F_BlockArr_int32_t_2_int32_t_F;

typedef struct __DistArray_int32_t_2__array_BlockArr_int32_t_2_int32_t_F_BlockArr_int32_t_2_int32_t_F* BlockDistArray2dIntChpl;

/**
 * END: Chapel implementation specific declarations
 */

/* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._hincludes) */

/*
 * Private data for class distarray.BlockDistArray2dInt
 */

struct distarray_BlockDistArray2dInt__data {
  /* DO-NOT-DELETE splicer.begin(distarray.BlockDistArray2dInt._data) */
  /* insert code here (private data members) */
  // TODO Reuse existing struct from babel for metadata?
  int dimension;
  int* lower;
  int* higher;
  BlockDistArray2dIntChpl chpl_data;
  /* DO-NOT-DELETE splicer.end(distarray.BlockDistArray2dInt._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct distarray_BlockDistArray2dInt__data*
distarray_BlockDistArray2dInt__get_data(
  distarray_BlockDistArray2dInt);

extern void
distarray_BlockDistArray2dInt__set_data(
  distarray_BlockDistArray2dInt,
  struct distarray_BlockDistArray2dInt__data*);

extern
void
impl_distarray_BlockDistArray2dInt__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__ctor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__ctor2(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt__dtor(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/
extern
void
impl_distarray_BlockDistArray2dInt_initArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t lo1,
  /* in */ int32_t hi1,
  /* in */ int32_t lo2,
  /* in */ int32_t hi2,
  /* in */ int32_t blk1,
  /* in */ int32_t blk2,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getDimension(
  /* in */ distarray_BlockDistArray2dInt self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getLower(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getHigher(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_distarray_BlockDistArray2dInt_getFromArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_distarray_BlockDistArray2dInt_setIntoArray(
  /* in */ distarray_BlockDistArray2dInt self,
  /* in */ int32_t newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ sidl_BaseInterface *_ex);

#ifdef WITH_RMI
extern struct sidl_BaseInterface__object* 
  impl_distarray_BlockDistArray2dInt_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
#endif /*WITH_RMI*/

/* DO-NOT-DELETE splicer.begin(_hmisc) */
/* insert code here (miscellaneous things) */
/* DO-NOT-DELETE splicer.end(_hmisc) */

#ifdef __cplusplus
}
#endif
#endif
