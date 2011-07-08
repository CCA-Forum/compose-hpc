/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_Stub.c
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Client-side glue code for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#define HPLSUPPORT_BLOCKCYCLICDISTARRAY2DDOUBLE_INLINE_DECL
#include "hplsupport_BlockCyclicDistArray2dDouble.h"
#include "hplsupport_BlockCyclicDistArray2dDouble_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#include "sidl_Exception.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

#define LANG_SPECIFIC_INIT()

/*
 * Hold pointer to IOR functions.
 */

static const struct hplsupport_BlockCyclicDistArray2dDouble__external 
  *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct hplsupport_BlockCyclicDistArray2dDouble__external* _loadIOR(
  void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = hplsupport_BlockCyclicDistArray2dDouble__externals();
#else
  _externals = (struct 
    hplsupport_BlockCyclicDistArray2dDouble__external*)sidl_dynamicLoadIOR(
    "hplsupport.BlockCyclicDistArray2dDouble",
    "hplsupport_BlockCyclicDistArray2dDouble__externals") ;
  sidl_checkIORVersion("hplsupport.BlockCyclicDistArray2dDouble", 
    _externals->d_ior_major_version, _externals->d_ior_minor_version, 2, 0);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct hplsupport_BlockCyclicDistArray2dDouble__sepv *_sepv = NULL;
/*
 * Return pointer to static functions.
 */

#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))
/*
 * Reset point to static functions.
 */

#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())

/*
 * Constructor function for the class.
 */

hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

/**
 * Wraps up the private data struct pointer (struct hplsupport_BlockCyclicDistArray2dDouble__data) passed in rather than running the constructor.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__wrapObj(void* data, 
  sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(data, _ex);
}

#ifdef WITH_RMI

static hplsupport_BlockCyclicDistArray2dDouble 
  hplsupport_BlockCyclicDistArray2dDouble__remoteCreate(const char* url, 
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__createRemote(const char* url, 
  sidl_BaseInterface *_ex)
{
  return hplsupport_BlockCyclicDistArray2dDouble__remoteCreate(url, _ex);
}

static struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  hplsupport_BlockCyclicDistArray2dDouble__remoteConnect(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
static struct hplsupport_BlockCyclicDistArray2dDouble__object* 
  hplsupport_BlockCyclicDistArray2dDouble__IHConnect(struct 
  sidl_rmi_InstanceHandle__object* instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__connect(const char* url, 
  sidl_BaseInterface *_ex)
{
  return hplsupport_BlockCyclicDistArray2dDouble__remoteConnect(url, TRUE, _ex);
}

#endif /*WITH_RMI*/

/*
 * Method to enable/disable static interface contract enforcement.
 */

void
hplsupport_BlockCyclicDistArray2dDouble__set_contracts_static(
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  (_getSEPV()->f__set_contracts_static)(
  enable, enfFilename, resetCounters, _ex);
  _resetSEPV();
}

/*
 * Method to dump static interface contract enforcement statistics.
 */

void
hplsupport_BlockCyclicDistArray2dDouble__dump_stats_static(
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  (_getSEPV()->f__dump_stats_static)(
  filename, prefix, _ex);
  _resetSEPV();
}

/*
 * Method to enable/disable interface contract enforcement.
 */

void
hplsupport_BlockCyclicDistArray2dDouble__set_contracts(
  hplsupport_BlockCyclicDistArray2dDouble self,
  sidl_bool   enable,
  const char* enfFilename,
  sidl_bool   resetCounters,
  struct sidl_BaseInterface__object **_ex)
{
  (*self->d_epv->f__set_contracts)(
  self,
  enable, enfFilename, resetCounters, _ex);
}

/*
 * Method to dump interface contract enforcement statistics.
 */

void
hplsupport_BlockCyclicDistArray2dDouble__dump_stats(
  hplsupport_BlockCyclicDistArray2dDouble self,
  const char* filename,
  const char* prefix,
  struct sidl_BaseInterface__object **_ex)
{
  (*self->d_epv->f__dump_stats)(
  self,
  filename, prefix, _ex);
}

/*
 * Method:  ptransHelper[]
 */

void
hplsupport_BlockCyclicDistArray2dDouble_ptransHelper(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble a,
  /* inout */ hplsupport_BlockCyclicDistArray2dDouble* c,
  /* in */ double beta,
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* out */ sidl_BaseInterface *_ex)
{
  (_getSEPV()->f_ptransHelper)(
    a,
    c,
    beta,
    i,
    j,
    _ex);
}

/*
 * Cast method for interface and class type conversions.
 */

hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  hplsupport_BlockCyclicDistArray2dDouble cast = NULL;

#ifdef WITH_RMI
  static int connect_loaded = 0;
  if (!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect(
      "hplsupport.BlockCyclicDistArray2dDouble", (
      void*)hplsupport_BlockCyclicDistArray2dDouble__IHConnect,_ex);SIDL_CHECK(
      *_ex);
  }
#endif /*WITH_RMI*/
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (hplsupport_BlockCyclicDistArray2dDouble) (*base->d_epv->f__cast)(
      base->d_object,
      "hplsupport.BlockCyclicDistArray2dDouble", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
hplsupport_BlockCyclicDistArray2dDouble__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface* _ex)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type, _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}




/*
 * TRUE if this object is remote, false if local
 */

sidl_bool
hplsupport_BlockCyclicDistArray2dDouble__isLocal(
  /* in */ hplsupport_BlockCyclicDistArray2dDouble self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !hplsupport_BlockCyclicDistArray2dDouble__isRemote(self, _ex);
}

/*
 * Method to enable/disable static hooks execution.
 */

void
hplsupport_BlockCyclicDistArray2dDouble__set_hooks_static(
  sidl_bool enable,
  struct sidl_BaseInterface__object **_ex)
{
  (_getSEPV()->f__set_hooks_static)(
  enable, _ex);
  _resetSEPV();
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_createCol
    (dimen, lower, upper);
}

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_createRow
    (dimen, lower, upper);
}

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_create1d(int32_t len)
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_create1d
    (len);
}

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_create1dInit(
  int32_t len, 
  hplsupport_BlockCyclicDistArray2dDouble* data)
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_create1dInit
    (len, (struct sidl_BaseInterface__object **)data);
}

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_create2dCol
    (m, n);
}

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_create2dRow
    (m, n);
}

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_borrow(
  hplsupport_BlockCyclicDistArray2dDouble* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct 
    hplsupport_BlockCyclicDistArray2dDouble__array*)sidl_interface__array_borrow
    (
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_smartCopy(
  struct hplsupport_BlockCyclicDistArray2dDouble__array *array)
{
  return (struct hplsupport_BlockCyclicDistArray2dDouble__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

/**
 * Increment the array's internal reference count by one.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_addRef(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_deleteRef(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get1(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get2(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get3(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get4(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get5(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get6(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get7(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
hplsupport_BlockCyclicDistArray2dDouble
hplsupport_BlockCyclicDistArray2dDouble__array_get(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t indices[])
{
  return (hplsupport_BlockCyclicDistArray2dDouble)
    sidl_interface__array_get((const struct sidl_interface__array *)array, 
      indices);
}

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set1(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set2(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set3(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set4(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set5(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set6(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set7(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
hplsupport_BlockCyclicDistArray2dDouble__array_set(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t indices[],
  hplsupport_BlockCyclicDistArray2dDouble const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices, (
    struct sidl_BaseInterface__object *)value);
}

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
hplsupport_BlockCyclicDistArray2dDouble__array_dimen(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hplsupport_BlockCyclicDistArray2dDouble__array_lower(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array, 
    ind);
}

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hplsupport_BlockCyclicDistArray2dDouble__array_upper(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array, 
    ind);
}

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hplsupport_BlockCyclicDistArray2dDouble__array_length(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array, 
    ind);
}

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
hplsupport_BlockCyclicDistArray2dDouble__array_stride(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array, 
    ind);
}

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
hplsupport_BlockCyclicDistArray2dDouble__array_isColumnOrder(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
hplsupport_BlockCyclicDistArray2dDouble__array_isRowOrder(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

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
hplsupport_BlockCyclicDistArray2dDouble__array_copy(
  const struct hplsupport_BlockCyclicDistArray2dDouble__array* src,
  struct hplsupport_BlockCyclicDistArray2dDouble__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

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
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_slice(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct hplsupport_BlockCyclicDistArray2dDouble__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

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
struct hplsupport_BlockCyclicDistArray2dDouble__array*
hplsupport_BlockCyclicDistArray2dDouble__array_ensure(
  struct hplsupport_BlockCyclicDistArray2dDouble__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct hplsupport_BlockCyclicDistArray2dDouble__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen, 
      ordering);
}

#ifdef WITH_RMI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif
#ifndef included_sidl_NotImplementedException_h
#include "sidl_NotImplementedException.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t hplsupport_BlockCyclicDistArray2dDouble__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &hplsupport_BlockCyclicDistArray2dDouble__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &hplsupport_BlockCyclicDistArray2dDouble__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &hplsupport_BlockCyclicDistArray2dDouble__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 2;
static const int32_t s_IOR_MINOR_VERSION = 0;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct hplsupport_BlockCyclicDistArray2dDouble__epv 
  s_rem_epv__hplsupport_blockcyclicdistarray2ddouble;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_hplsupport_BlockCyclicDistArray2dDouble__cast(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int cmp;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp = strcmp(name, "sidl.BaseClass");
  if (!cmp) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidl_BaseClass__object*)self);
    return cast;
  }
  else if (cmp < 0) {
    cmp = strcmp(name, "hplsupport.BlockCyclicDistArray2dDouble");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct hplsupport_BlockCyclicDistArray2dDouble__object*)self);
      return cast;
    }
  }
  else if (cmp > 0) {
    cmp = strcmp(name, "sidl.BaseInterface");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
      sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih, 
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_hplsupport_BlockCyclicDistArray2dDouble__delete(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
  struct sidl_BaseInterface__object* *_ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_hplsupport_BlockCyclicDistArray2dDouble__getURL(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self, struct 
    sidl_BaseInterface__object* *_ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_hplsupport_BlockCyclicDistArray2dDouble__raddRef(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self,struct 
    sidl_BaseInterface__object* *_ex)
{
  struct sidl_BaseException__object* netex = NULL;
  /* initialize a new invocation */
  struct sidl_BaseInterface__object* _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    *_ex = (struct sidl_BaseInterface__object*)netex;
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_hplsupport_BlockCyclicDistArray2dDouble__isRemote(
    struct hplsupport_BlockCyclicDistArray2dDouble__object* self, 
    struct sidl_BaseInterface__object* *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble__set_hooks(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ sidl_bool enable,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble._set_hooks.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* Contract enforcement has not been implemented for remote use. */
/* REMOTE METHOD STUB:_set_contracts */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble__set_contracts(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ sidl_bool enable,
  /* in */ const char* enfFilename,
  /* in */ sidl_bool resetCounters,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_contracts", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "enfFilename", enfFilename, 
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packBool( _inv, "resetCounters", resetCounters, 
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble._set_contracts.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* Contract enforcement has not been implemented for remote use. */
/* REMOTE METHOD STUB:_dump_stats */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble__dump_stats(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ const char* filename,
  /* in */ const char* prefix,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_dump_stats", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex);SIDL_CHECK(
      *_ex);
    sidl_rmi_Invocation_packString( _inv, "prefix", prefix, _ex);SIDL_CHECK(
      *_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble._dump_stats.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_hplsupport_BlockCyclicDistArray2dDouble__exec(
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self,const char* 
    methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  struct sidl_BaseInterface__object* *_ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:initData */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble_initData(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ void* data,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "initData", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packOpaque( _inv, "data", data, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.initData.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:get */
static double
remote_hplsupport_BlockCyclicDistArray2dDouble_get(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    double _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "get", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "idx1", idx1, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "idx2", idx2, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.get.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackDouble( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:set */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble_set(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ double newVal,
  /* in */ int32_t idx1,
  /* in */ int32_t idx2,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "set", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "newVal", newVal, _ex);SIDL_CHECK(
      *_ex);
    sidl_rmi_Invocation_packInt( _inv, "idx1", idx1, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "idx2", idx2, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.set.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:addRef */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble_addRef(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct hplsupport_BlockCyclicDistArray2dDouble__remote* r_obj = (struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
#ifdef SIDL_DEBUG_REFCOUNT
    fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
      r_obj, r_obj->d_refcount, 
      "hplsupport.BlockCyclicDistArray2dDouble Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_hplsupport_BlockCyclicDistArray2dDouble_deleteRef(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct hplsupport_BlockCyclicDistArray2dDouble__remote* r_obj = (struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
#ifdef SIDL_DEBUG_REFCOUNT
    fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",r_obj, r_obj->d_refcount, "hplsupport.BlockCyclicDistArray2dDouble Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_hplsupport_BlockCyclicDistArray2dDouble_isSame(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.isSame.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_hplsupport_BlockCyclicDistArray2dDouble_isType(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.isType.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_hplsupport_BlockCyclicDistArray2dDouble_getClassInfo(
  /* in */ struct hplsupport_BlockCyclicDistArray2dDouble__object*self ,
  /* out */ struct sidl_BaseInterface__object **_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    struct sidl_BaseInterface__object* _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      hplsupport_BlockCyclicDistArray2dDouble__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if (_be != NULL) {
      struct sidl_BaseInterface__object* throwaway_exception = NULL;
      sidl_BaseException_addLine(_be, 
      "Exception unserialized from hplsupport.BlockCyclicDistArray2dDouble.getClassInfo.",
        &throwaway_exception);
      *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void hplsupport_BlockCyclicDistArray2dDouble__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct hplsupport_BlockCyclicDistArray2dDouble__epv* epv = 
    &s_rem_epv__hplsupport_blockcyclicdistarray2ddouble;
  struct sidl_BaseClass__epv*                          e0  = 
    &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*                      e1  = 
    &s_rem_epv__sidl_baseinterface;

  epv->f__cast               = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__cast;
  epv->f__delete             = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__delete;
  epv->f__exec               = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__exec;
  epv->f__getURL             = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__getURL;
  epv->f__raddRef            = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__raddRef;
  epv->f__isRemote           = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__isRemote;
  epv->f__set_hooks          = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__set_hooks;
  epv->f__set_contracts      = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__set_contracts;
  epv->f__dump_stats         = 
    remote_hplsupport_BlockCyclicDistArray2dDouble__dump_stats;
  epv->f__ctor               = NULL;
  epv->f__ctor2              = NULL;
  epv->f__dtor               = NULL;
  epv->f_initData            = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_initData;
  epv->f_get        = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_get;
  epv->f_set        = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_set;
  epv->f_addRef              = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_addRef;
  epv->f_deleteRef           = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_deleteRef;
  epv->f_isSame              = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_isSame;
  epv->f_isType              = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_isType;
  epv->f_getClassInfo        = 
    remote_hplsupport_BlockCyclicDistArray2dDouble_getClassInfo;

  e0->f__cast          = (void* (*)(struct sidl_BaseClass__object*, const char*,
    struct sidl_BaseInterface__object**)) epv->f__cast;
  e0->f__delete        = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__delete;
  e0->f__getURL        = (char* (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__getURL;
  e0->f__raddRef       = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__raddRef;
  e0->f__isRemote      = (sidl_bool (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object**)) epv->f__isRemote;
  e0->f__set_hooks     = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
    struct sidl_BaseInterface__object**)) epv->f__set_hooks;
  e0->f__set_contracts = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
    const char*, sidl_bool, struct sidl_BaseInterface__object**)) 
    epv->f__set_contracts;
  e0->f__dump_stats    = (void (*)(struct sidl_BaseClass__object*, const char*, 
    const char*, struct sidl_BaseInterface__object**)) epv->f__dump_stats;
  e0->f__exec          = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef         = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef      = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame         = (sidl_bool (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e0->f_isType         = (sidl_bool (*)(struct sidl_BaseClass__object*,const 
    char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;

  e1->f__cast          = (void* (*)(void*, const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e1->f__delete        = (void (*)(void*, struct sidl_BaseInterface__object**)) 
    epv->f__delete;
  e1->f__getURL        = (char* (*)(void*, struct 
    sidl_BaseInterface__object**)) epv->f__getURL;
  e1->f__raddRef       = (void (*)(void*, struct sidl_BaseInterface__object**)) 
    epv->f__raddRef;
  e1->f__isRemote      = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object**)) epv->f__isRemote;
  e1->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
    sidl_BaseInterface__object**)) epv->f__set_hooks;
  e1->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
    struct sidl_BaseInterface__object**)) epv->f__set_contracts;
  e1->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
    sidl_BaseInterface__object**)) epv->f__dump_stats;
  e1->f__exec          = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_addRef;
  e1->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_deleteRef;
  e1->f_isSame         = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e1->f_isType         = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct hplsupport_BlockCyclicDistArray2dDouble__object*
hplsupport_BlockCyclicDistArray2dDouble__remoteConnect(const char *url, 
  sidl_bool ar, struct sidl_BaseInterface__object* *_ex)
{
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self = NULL;

  struct hplsupport_BlockCyclicDistArray2dDouble__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hplsupport_BlockCyclicDistArray2dDouble__remote* r_obj = NULL;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    struct hplsupport_BlockCyclicDistArray2dDouble__object* retobj = NULL;
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_BaseInterface bi = (
      sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(objectID,
      _ex); SIDL_CHECK(*_ex);
    (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
    retobj = (struct hplsupport_BlockCyclicDistArray2dDouble__object*) (
      *bi->d_epv->f__cast)(bi->d_object, 
      "hplsupport.BlockCyclicDistArray2dDouble", _ex);
    if(!ar) { 
      (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
    }
    return retobj;
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, 
    "hplsupport.BlockCyclicDistArray2dDouble", ar, _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct hplsupport_BlockCyclicDistArray2dDouble__object*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__object));

  r_obj =
    (struct hplsupport_BlockCyclicDistArray2dDouble__remote*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hplsupport.BlockCyclicDistArray2dDouble.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                                   self;
  s1 =                                                   &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hplsupport_BlockCyclicDistArray2dDouble__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hplsupport_blockcyclicdistarray2ddouble;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct hplsupport_BlockCyclicDistArray2dDouble__object*
hplsupport_BlockCyclicDistArray2dDouble__IHConnect(sidl_rmi_InstanceHandle 
  instance, struct sidl_BaseInterface__object* *_ex)
{
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self = NULL;

  struct hplsupport_BlockCyclicDistArray2dDouble__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hplsupport_BlockCyclicDistArray2dDouble__remote* r_obj = NULL;
  self =
    (struct hplsupport_BlockCyclicDistArray2dDouble__object*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__object));

  r_obj =
    (struct hplsupport_BlockCyclicDistArray2dDouble__remote*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hplsupport.BlockCyclicDistArray2dDouble.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                                   self;
  s1 =                                                   &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hplsupport_BlockCyclicDistArray2dDouble__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hplsupport_blockcyclicdistarray2ddouble;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct hplsupport_BlockCyclicDistArray2dDouble__object*
hplsupport_BlockCyclicDistArray2dDouble__remoteCreate(const char *url, struct 
  sidl_BaseInterface__object **_ex)
{
  struct sidl_BaseInterface__object* _throwaway_exception = NULL;
  struct hplsupport_BlockCyclicDistArray2dDouble__object* self = NULL;

  struct hplsupport_BlockCyclicDistArray2dDouble__object* s0;
  struct sidl_BaseClass__object* s1;

  struct hplsupport_BlockCyclicDistArray2dDouble__remote* r_obj = NULL;
  sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
    url, "hplsupport.BlockCyclicDistArray2dDouble", _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct hplsupport_BlockCyclicDistArray2dDouble__object*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__object));

  r_obj =
    (struct hplsupport_BlockCyclicDistArray2dDouble__remote*) malloc(
      sizeof(struct hplsupport_BlockCyclicDistArray2dDouble__remote));

  if(!self || !r_obj) {
    sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
      _ex);
    SIDL_CHECK(*_ex);
    sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(*_ex);
    sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
      "hplsupport.BlockCyclicDistArray2dDouble.EPVgeneration", _ex);
    SIDL_CHECK(*_ex);
    *_ex = (struct sidl_BaseInterface__object*)ex;
    goto EXIT;
  }

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                                   self;
  s1 =                                                   &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    hplsupport_BlockCyclicDistArray2dDouble__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__hplsupport_blockcyclicdistarray2ddouble;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance, 
    &_throwaway_exception); }
  if(self) { free(self); }
  if(r_obj) { free(r_obj); }
  return NULL;
}
/*
 * RMI connector function for the class.
 */

struct hplsupport_BlockCyclicDistArray2dDouble__object*
hplsupport_BlockCyclicDistArray2dDouble__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return hplsupport_BlockCyclicDistArray2dDouble__remoteConnect(url, ar, _ex);
}


#endif /*WITH_RMI*/
