#ifndef __BLAS_VECTORUTILS_CSTUB_H__
#define __BLAS_VECTORUTILS_CSTUB_H__
#include <blas_VectorUtils_IOR.h>
#include <blas_VectorUtils_IOR.h>
#include <sidlType.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>
void blas_VectorUtils_helper_daxpy_stub( int32_t n, double alpha, 
  sidl_double__array X, sidl_double__array Y, struct sidl_BaseInterface__object** 
  ex);

#endif
