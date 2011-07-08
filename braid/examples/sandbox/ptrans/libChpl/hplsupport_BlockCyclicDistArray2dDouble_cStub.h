#ifndef __HPLSUPPORT_BLOCKCYCLICDISTARRAY2DDOUBLE_CSTUB_H__
#define __HPLSUPPORT_BLOCKCYCLICDISTARRAY2DDOUBLE_CSTUB_H__
#include <hplsupport_BlockCyclicDistArray2dDouble_IOR.h>
#include <hplsupport_BlockCyclicDistArray2dDouble_IOR.h>
#include <sidlType.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>
void hplsupport_BlockCyclicDistArray2dDouble_initData_stub( struct 
  hplsupport_BlockCyclicDistArray2dDouble__object* self, void* data, struct 
  sidl_BaseInterface__object** ex);
double hplsupport_BlockCyclicDistArray2dDouble_get_stub( struct 
  hplsupport_BlockCyclicDistArray2dDouble__object* self, int32_t idx1, int32_t 
  idx2, struct sidl_BaseInterface__object** ex);
void hplsupport_BlockCyclicDistArray2dDouble_set_stub( struct 
  hplsupport_BlockCyclicDistArray2dDouble__object* self, double newVal, int32_t 
  idx1, int32_t idx2, struct sidl_BaseInterface__object** ex);
void hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( struct 
  hplsupport_BlockCyclicDistArray2dDouble__object* a, struct 
  hplsupport_BlockCyclicDistArray2dDouble__object** c, double beta, int32_t i, 
  int32_t j, struct sidl_BaseInterface__object** ex);

#endif
