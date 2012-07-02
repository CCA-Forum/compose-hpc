#include <pgas_blockedDoubleArray_IOR.h>
#include <pgas_blockedDoubleArray_Skel.h>
#include <stdint.h>
/**
 * builtin method
 */
void pgas_blockedDoubleArray__ctor_skel( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray__ctor_impl( self->d_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDoubleArray__ctor2_skel( struct pgas_blockedDoubleArray__object* self, void* private_data, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray__ctor2_impl( self->d_data,   private_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDoubleArray__dtor_skel( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray__dtor_impl( self->d_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDoubleArray__load_skel( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray__load_impl( self->d_data,   _ex);
}

void pgas_blockedDoubleArray_allocate_skel( struct pgas_blockedDoubleArray__object* self, int size, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray_allocate_impl( self->d_data,   size,   _ex);
}

double pgas_blockedDoubleArray_get_skel( struct pgas_blockedDoubleArray__object* self, int idx, struct sidl_BaseInterface__object** _ex) {
double _retval;
  return pgas_blockedDoubleArray_get_impl( self->d_data,   idx,   _ex);
}

void pgas_blockedDoubleArray_set_skel( struct pgas_blockedDoubleArray__object* self, int idx, double val, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDoubleArray_set_impl( self->d_data,   idx,   val,   _ex);
}

void pgas_blockedDoubleArray__call_load() {
  /* FIXME: [ir.Stmt(ir.Call('_load', [])) */
}

void pgas_blockedDoubleArray__set_epv( struct pgas_blockedDoubleArray__epv* epv, struct pgas_blockedDoubleArray__pre_epv* pre_epv, struct pgas_blockedDoubleArray__post_epv* post_epv) {
  epv->f__ctor = pgas_blockedDoubleArray__ctor_skel;
  epv->f__ctor2 = pgas_blockedDoubleArray__ctor2_skel;
  epv->f__dtor = pgas_blockedDoubleArray__dtor_skel;
  epv->f__load = pgas_blockedDoubleArray__load_skel;
  epv->f_allocate = pgas_blockedDoubleArray_allocate_skel;
  epv->f_get = pgas_blockedDoubleArray_get_skel;
  epv->f_set = pgas_blockedDoubleArray_set_skel;
  const char* name[] = { "BRAID_LIBRARY", "-v" }; // verbose Chapel;
  chpl_init_library(2, &name);
  chpl__init_chpl__Program(__LINE__, __FILE__);
  chpl__init_pgas_Impl(__LINE__, __FILE__);
}
