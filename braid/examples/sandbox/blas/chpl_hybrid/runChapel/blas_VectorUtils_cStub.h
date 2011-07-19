#ifndef __BLAS_VECTORUTILS_CSTUB_H__
#define __BLAS_VECTORUTILS_CSTUB_H__
#include <blas_VectorUtils_IOR.h>
#include <blas_VectorUtils_IOR.h>
#include <sidlType.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>
void blas_VectorUtils_helper_daxpy_stub( int32_t n, double alpha, 
  sidl_double__array X, sidl_double__array Y, struct sidl_BaseInterface__object** 
  _ex);
/**
 * Implicit built-in method: addRef
 */
void blas_VectorUtils_addRef_stub( struct blas_VectorUtils__object* self, struct sidl_BaseInterface__object** _ex);
/**
 * Implicit built-in method: deleteRef
 */
void blas_VectorUtils_deleteRef_stub( struct blas_VectorUtils__object* self, struct sidl_BaseInterface__object** _ex);

#endif
