#include <pgas_blockedDouble3dArray_IOR.h>
#include <pgas_blockedDouble3dArray_Skel.h>
#include <stdint.h>
/**
 * builtin method
 */
void pgas_blockedDouble3dArray__ctor_skel( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray__ctor_impl( self->d_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDouble3dArray__ctor2_skel( struct pgas_blockedDouble3dArray__object* self, void* private_data, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray__ctor2_impl( self->d_data,   private_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDouble3dArray__dtor_skel( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray__dtor_impl( self->d_data,   _ex);
}

/**
 * builtin method
 */
void pgas_blockedDouble3dArray__load_skel( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray__load_impl( self->d_data,   _ex);
}

/**
 * 
 * allocate a blocked cubic array of doubles in size*size*size
 * 
 */
void pgas_blockedDouble3dArray_allocate_skel( struct pgas_blockedDouble3dArray__object* self, int size, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray_allocate_impl( self->d_data,   size,   _ex);
}

double pgas_blockedDouble3dArray_get_skel( struct pgas_blockedDouble3dArray__object* self, int idx1, int idx2, int idx3, struct sidl_BaseInterface__object** _ex) {
double _retval;
  return pgas_blockedDouble3dArray_get_impl( self->d_data,   idx1,   idx2,   idx3,   _ex);
}

void pgas_blockedDouble3dArray_set_skel( struct pgas_blockedDouble3dArray__object* self, int idx1, int idx2, int idx3, double val, struct sidl_BaseInterface__object** _ex) {
  pgas_blockedDouble3dArray_set_impl( self->d_data,   idx1,   idx2,   idx3,   val,   _ex);
}

void pgas_blockedDouble3dArray__call_load() {
  /* FIXME: [ir.Stmt(ir.Call('_load', [])) */
}

void pgas_blockedDouble3dArray__set_epv( struct pgas_blockedDouble3dArray__epv* epv, struct pgas_blockedDouble3dArray__pre_epv* pre_epv, struct pgas_blockedDouble3dArray__post_epv* post_epv) {
  epv->f__ctor = pgas_blockedDouble3dArray__ctor_skel;
  epv->f__ctor2 = pgas_blockedDouble3dArray__ctor2_skel;
  epv->f__dtor = pgas_blockedDouble3dArray__dtor_skel;
  epv->f__load = pgas_blockedDouble3dArray__load_skel;
  epv->f_allocate = pgas_blockedDouble3dArray_allocate_skel;
  epv->f_get = pgas_blockedDouble3dArray_get_skel;
  epv->f_set = pgas_blockedDouble3dArray_set_skel;
  const char* name[] = { "BRAID_LIBRARY", "-v" }; // verbose Chapel;
  chpl_init_library(2, &name);
  chpl__init_chpl__Program(__LINE__, __FILE__);
  chpl__init_pgas_Impl(__LINE__, __FILE__);
}
