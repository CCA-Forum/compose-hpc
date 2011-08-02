#ifndef __BLAS_VECTORUTILS_STUB_H__
#define __BLAS_VECTORUTILS_STUB_H__
// Package header (enums, etc...)
#include <stdint.h>
#include <blas.h>
#include <blas_VectorUtils_IOR.h>
typedef struct blas_VectorUtils__object _blas_VectorUtils__object;
typedef _blas_VectorUtils__object* blas_VectorUtils__object;
#ifndef SIDL_BASE_INTERFACE_OBJECT
#define SIDL_BASE_INTERFACE_OBJECT
typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;
typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;
#define IS_NULL(aPtr) ((*aPtr) != NULL)
#endif
blas_VectorUtils__object blas_VectorUtils__createObject(blas_VectorUtils__object copy, sidl_BaseInterface__object* ex);

#endif
