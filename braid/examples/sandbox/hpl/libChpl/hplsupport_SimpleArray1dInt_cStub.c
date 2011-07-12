#include <hplsupport_SimpleArray1dInt_cStub.h>

void hplsupport_SimpleArray1dInt_initData_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		void* data,
		struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_initData)( self, data, ex);
}

int32_t hplsupport_SimpleArray1dInt_get_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		int32_t idx1,
		struct sidl_BaseInterface__object** ex) {
  int _retval;
_retval = (*self->d_epv->f_get)( self, idx1, ex);
  return _retval;
}

void hplsupport_SimpleArray1dInt_set_stub(
		struct hplsupport_SimpleArray1dInt__object* self,
		int32_t newVal,
		int32_t idx1,
		struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_set)( self, newVal, idx1, ex);
}
