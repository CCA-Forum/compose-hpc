#ifndef __HPLSUPPORT_SIMPLEARRAY1DINT_CSTUB_H__
#define __HPLSUPPORT_SIMPLEARRAY1DINT_CSTUB_H__

#include <hplsupport_SimpleArray1dInt_IOR.h>

#include <sidlType.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>

void hplsupport_SimpleArray1dInt_initData_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		void* data,
		struct sidl_BaseInterface__object** ex);


int32_t hplsupport_SimpleArray1dInt_get_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		int32_t idx1,
		struct sidl_BaseInterface__object** ex);


void hplsupport_SimpleArray1dInt_set_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		int32_t newVal,
		int32_t idx1,
		struct sidl_BaseInterface__object** ex);

#endif
