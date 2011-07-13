#include <hplsupport_BlockCyclicDistArray2dDouble_cStub.h>

void hplsupport_BlockCyclicDistArray2dDouble_initData_stub(
		struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
		void* data,
		struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_initData)( self, data, ex);
}

double hplsupport_BlockCyclicDistArray2dDouble_get_stub(
		struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
		int32_t idx1,
		int32_t idx2,
		struct sidl_BaseInterface__object** ex) {
  double _retval;
_retval = (*self->d_epv->f_get)( self, idx1, idx2, ex);
  return _retval;
}

void hplsupport_BlockCyclicDistArray2dDouble_set_stub(
		struct hplsupport_BlockCyclicDistArray2dDouble__object* self,
		double newVal,
		int32_t idx1,
		int32_t idx2,
		struct sidl_BaseInterface__object** ex) {
  (*self->d_epv->f_set)( self, newVal, idx1, idx2, ex);
}
