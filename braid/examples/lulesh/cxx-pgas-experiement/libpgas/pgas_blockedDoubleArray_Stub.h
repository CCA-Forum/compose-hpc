#ifndef included_pgas_blockedDoubleArray_Stub_h
#define included_pgas_blockedDoubleArray_Stub_h
// Class header
#include <stdint.h>
#include <pgas.h>
#include <pgas_blockedDoubleArray_IOR.h>
typedef struct pgas_blockedDoubleArray__object _pgas_blockedDoubleArray__object;
typedef _pgas_blockedDoubleArray__object* pgas_blockedDoubleArray__object;
#ifndef SIDL_BASE_INTERFACE_OBJECT
#define SIDL_BASE_INTERFACE_OBJECT
typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;
typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;
#define IS_NULL(aPtr)     ((aPtr) == 0)
#define IS_NOT_NULL(aPtr) ((aPtr) != 0)
#define SET_TO_NULL(aPtr) ((*aPtr) = 0)
#endif
#define _cast_sidl_BaseClass(ior,ex) ((struct \
  sidl_BaseClass__object*)((*ior->d_epv->f__cast)(ior,"sidl.BaseClass",ex)))
#define sidl_BaseClass_cast_pgas_blockedDoubleArray(ior) ((struct \
  sidl_BaseClass__object*)((struct sidl_BaseInterface__object*)ior)->d_object)
#define _cast_sidl_BaseInterface(ior,ex) ((struct \
  sidl_BaseInterface__object*)((*ior->d_epv->f__cast)(ior,"sidl.BaseInterface",ex)))
#define sidl_BaseInterface_cast_pgas_blockedDoubleArray(ior) ((struct \
  sidl_BaseInterface__object*)((struct \
  sidl_BaseInterface__object*)ior)->d_object)
#endif

