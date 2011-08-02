/*
 * File:          hpcc_HighPerformanceLinpack.h
 * Symbol:        hpcc.HighPerformanceLinpack-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Client-side glue code for hpcc.HighPerformanceLinpack
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_hpcc_HighPerformanceLinpack_h
#define included_hpcc_HighPerformanceLinpack_h

/**
 * Symbol "hpcc.HighPerformanceLinpack" (version 0.1)
 */
struct hpcc_HighPerformanceLinpack__object;
struct hpcc_HighPerformanceLinpack__array;
typedef struct hpcc_HighPerformanceLinpack__object* hpcc_HighPerformanceLinpack;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_h
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#endif
#ifndef included_hplsupport_SimpleArray1dInt_h
#include "hplsupport_SimpleArray1dInt.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
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
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_hpcc_HighPerformanceLinpack_IOR_h
#include "hpcc_HighPerformanceLinpack_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

#if !defined(HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL) && defined(SIDL_C_HAS_INLINE)
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || defined(__INTEL_COMPILER) || !defined(__GNUC__)
#define HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL inline
#else
#define HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL extern inline
#endif
#endif /* inline preprocessor definitions */
/**
 * Constructor function for the class.
 */
struct hpcc_HighPerformanceLinpack__object*
hpcc_HighPerformanceLinpack__create(sidl_BaseInterface* _ex);

/**
 * Wraps up the private data struct pointer (struct hpcc_HighPerformanceLinpack__data) passed in rather than running the constructor.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__wrapObj(void * data, sidl_BaseInterface *_ex);

#ifdef WITH_RMI

/**
 * RMI constructor function for the class.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__createRemote(const char * url, sidl_BaseInterface 
  *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__connect(const char *, sidl_BaseInterface *_ex);


#endif /*WITH_RMI*/
/**
 * Method to enable/disable static interface contract enforcement.
 */
void
hpcc_HighPerformanceLinpack__set_contracts_static(
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex);

/**
 * Method to dump static interface contract enforcement statistics.
 */
void
hpcc_HighPerformanceLinpack__dump_stats_static(
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex);

/**
 * Method to enable/disable interface contract enforcement.
 */
void
hpcc_HighPerformanceLinpack__set_contracts(
  hpcc_HighPerformanceLinpack self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex);

/**
 * Method to dump interface contract enforcement statistics.
 */
void
hpcc_HighPerformanceLinpack__dump_stats(
  hpcc_HighPerformanceLinpack self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex);

/**
 * Method:  panelSolveCompute[]
 */
void
hpcc_HighPerformanceLinpack_panelSolveCompute(
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

/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
void
hpcc_HighPerformanceLinpack_addRef(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */


/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
void
hpcc_HighPerformanceLinpack_deleteRef(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */


/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
sidl_bool
hpcc_HighPerformanceLinpack_isSame(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
  return _result;
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */


/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
sidl_bool
hpcc_HighPerformanceLinpack_isType(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
  return _result;
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */


/**
 * Return the meta-data about the class implementing this interface.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
sidl_ClassInfo
hpcc_HighPerformanceLinpack_getClassInfo(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  sidl_ClassInfo _result;
  _result = (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */


/**
 * Cast method for interface and class type conversions.
 */
struct hpcc_HighPerformanceLinpack__object*
hpcc_HighPerformanceLinpack__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
hpcc_HighPerformanceLinpack__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
void
hpcc_HighPerformanceLinpack__exec(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  (*self->d_epv->f__exec)(
    self,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */

/**
 * Get the URL of the Implementation of this object (for RMI)
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
char*
hpcc_HighPerformanceLinpack__getURL(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  char* _result;
  _result = (*self->d_epv->f__getURL)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */

/**
 * On a remote object, addrefs the remote instance.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
void
hpcc_HighPerformanceLinpack__raddRef(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  (*self->d_epv->f__raddRef)(
    self,
    _ex);
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */

/**
 * TRUE if this object is remote, false if local
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
sidl_bool
hpcc_HighPerformanceLinpack__isRemote(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  sidl_bool _result;
  _result = (*self->d_epv->f__isRemote)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
hpcc_HighPerformanceLinpack__isLocal(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* out */ sidl_BaseInterface *_ex);
/**
 * Static Method to enable/disable method hooks invocation.
 */
void
hpcc_HighPerformanceLinpack__set_hooks_static(
  /* in */ sidl_bool enable,
  /* out */ sidl_BaseInterface *_ex);
/**
 * Method to enable/disable method hooks invocation.
 */
HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
void
hpcc_HighPerformanceLinpack__set_hooks(
  /* in */ hpcc_HighPerformanceLinpack self,
  /* in */ sidl_bool enable,
  /* out */ sidl_BaseInterface *_ex)
#ifdef HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL
{
  (*self->d_epv->f__set_hooks)(
    self,
    enable,
    _ex);
}
#else
;
#endif /* HPCC_HIGHPERFORMANCELINPACK_INLINE_DECL */

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_create1d(int32_t len);

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_create1dInit(
  int32_t len, 
  hpcc_HighPerformanceLinpack* data);

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_borrow(
  hpcc_HighPerformanceLinpack* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_smartCopy(
  struct hpcc_HighPerformanceLinpack__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
hpcc_HighPerformanceLinpack__array_addRef(
  struct hpcc_HighPerformanceLinpack__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
hpcc_HighPerformanceLinpack__array_deleteRef(
  struct hpcc_HighPerformanceLinpack__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get1(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get2(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get3(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get4(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get5(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get6(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get7(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
hpcc_HighPerformanceLinpack
hpcc_HighPerformanceLinpack__array_get(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set1(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set2(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set3(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set4(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set5(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set6(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
hpcc_HighPerformanceLinpack__array_set7(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  hpcc_HighPerformanceLinpack const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
hpcc_HighPerformanceLinpack__array_set(
  struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t indices[],
  hpcc_HighPerformanceLinpack const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
hpcc_HighPerformanceLinpack__array_dimen(
  const struct hpcc_HighPerformanceLinpack__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hpcc_HighPerformanceLinpack__array_lower(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hpcc_HighPerformanceLinpack__array_upper(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t ind);

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hpcc_HighPerformanceLinpack__array_length(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hpcc_HighPerformanceLinpack__array_stride(
  const struct hpcc_HighPerformanceLinpack__array* array,
  const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
hpcc_HighPerformanceLinpack__array_isColumnOrder(
  const struct hpcc_HighPerformanceLinpack__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
hpcc_HighPerformanceLinpack__array_isRowOrder(
  const struct hpcc_HighPerformanceLinpack__array* array);

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_slice(
  struct hpcc_HighPerformanceLinpack__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
hpcc_HighPerformanceLinpack__array_copy(
  const struct hpcc_HighPerformanceLinpack__array* src,
  struct hpcc_HighPerformanceLinpack__array* dest);

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct hpcc_HighPerformanceLinpack__array*
hpcc_HighPerformanceLinpack__array_ensure(
  struct hpcc_HighPerformanceLinpack__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak hpcc_HighPerformanceLinpack__connectI

/**
 * RMI connector function for the class. (no addref)
 */
struct hpcc_HighPerformanceLinpack__object*
hpcc_HighPerformanceLinpack__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
