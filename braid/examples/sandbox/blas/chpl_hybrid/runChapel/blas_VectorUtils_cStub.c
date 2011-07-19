#include <blas_VectorUtils_cStub.h>

// Hold pointer to IOR functions.
static const struct blas_VectorUtils__external *_externals = NULL;

extern const struct blas_VectorUtils__external* blas_VectorUtils__externals();

// Lookup the symbol to get the IOR functions.
static const struct blas_VectorUtils__external* _loadIOR(void)

// Return pointer to internal IOR functions.
{
#ifdef SIDL_STATIC_LIBRARY
  _externals = blas_VectorUtils__externals();
#else
  _externals = (struct blas_VectorUtils__external*)sidl_dynamicLoadIOR(
    "ArrayTest.ArrayOps","blas_VectorUtils__externals") ;
  sidl_checkIORVersion("blas.VectorUtils", _externals->d_ior_major_version, 
    _externals->d_ior_minor_version, 2, 0);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

// Hold pointer to static entry point vector
static const struct blas_VectorUtils__sepv *_sepv = NULL;

// Return pointer to static functions.
#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))

// Reset point to static functions.
#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())


void blas_VectorUtils_helper_daxpy_stub( int32_t n, double alpha, sidl_double__array X, sidl_double__array Y, struct sidl_BaseInterface__object** _ex) {
  (_getSEPV()->f_helper_daxpy)( n, alpha, X, Y, _ex);
}

/**
 * Implicit built-in method: addRef
 */
void blas_VectorUtils_addRef_stub( struct blas_VectorUtils__object* self, struct sidl_BaseInterface__object** _ex) {
  (*self->d_epv->f_addRef)( self, _ex);
}

/**
 * Implicit built-in method: deleteRef
 */
void blas_VectorUtils_deleteRef_stub( struct blas_VectorUtils__object* self, struct sidl_BaseInterface__object** _ex) {
  (*self->d_epv->f_deleteRef)( self, _ex);
}
