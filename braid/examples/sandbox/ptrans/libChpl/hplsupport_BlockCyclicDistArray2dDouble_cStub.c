#include <hplsupport_BlockCyclicDistArray2dDouble_cStub.h>

// Hold pointer to IOR functions.
static const struct hplsupport_BlockCyclicDistArray2dDouble__external *_externals = NULL;

// Lookup the symbol to get the IOR functions.
static const struct hplsupport_BlockCyclicDistArray2dDouble__external* _loadIOR(void)

// Return pointer to internal IOR functions.
{
#ifdef SIDL_STATIC_LIBRARY
  _externals = hplsupport_BlockCyclicDistArray2dDouble__externals();
#else
  _externals = (struct hplsupport_BlockCyclicDistArray2dDouble__external*)sidl_dynamicLoadIOR(
    "ArrayTest.ArrayOps","hplsupport_BlockCyclicDistArray2dDouble__externals") ;
  sidl_checkIORVersion("hplsupport.BlockCyclicDistArray2dDouble", _externals->d_ior_major_version, 
    _externals->d_ior_minor_version, 2, 0);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

// Hold pointer to static entry point vector
static const struct hplsupport_BlockCyclicDistArray2dDouble__sepv *_sepv = NULL;

// Return pointer to static functions.
#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))

// Reset point to static functions.
#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())


void hplsupport_BlockCyclicDistArray2dDouble_initData_stub( struct hplsupport_BlockCyclicDistArray2dDouble__object* self, void* data, struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_initData)( self, data, ex);
}

double hplsupport_BlockCyclicDistArray2dDouble_get_stub( struct hplsupport_BlockCyclicDistArray2dDouble__object* self, int32_t idx1, int32_t idx2, struct sidl_BaseInterface__object** ex) {
  double _retval;
_retval = (*self->d_epv->f_get)( self, idx1, idx2, ex);
  return _retval;
}

void hplsupport_BlockCyclicDistArray2dDouble_set_stub( struct hplsupport_BlockCyclicDistArray2dDouble__object* self, double newVal, int32_t idx1, int32_t idx2, struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_set)( self, newVal, idx1, idx2, ex);
}

void hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( struct hplsupport_BlockCyclicDistArray2dDouble__object* a, struct hplsupport_BlockCyclicDistArray2dDouble__object* c, double beta, int32_t i, int32_t j, struct sidl_BaseInterface__object** ex) {
  (_getSEPV()->f_ptransHelper)( a, c, beta, i, j, ex);
}
