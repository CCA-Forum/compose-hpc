#ifndef __BLAS_VECTORUTILS_STUB_H__
#define __BLAS_VECTORUTILS_STUB_H__
// Package header (enums, etc...)
#include <stdint.h>
#include <blas.h>
#include <blas_VectorUtils_IOR.h>
typedef struct blas_VectorUtils__object _blas_VectorUtils__object;
typedef _blas_VectorUtils__object* blas_VectorUtils__object;
#ifndef _CHPL_SIDL_BASETYPES
#define _CHPL_SIDL_BASETYPES
typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;
typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;
blas_VectorUtils__object blas_VectorUtils__createObject(blas_VectorUtils__object copy, sidl_BaseInterface__object* ex);
#endif

#endif
