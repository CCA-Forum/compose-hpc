#ifndef included_pgas_blockedDoubleArray_Skel_h
#define included_pgas_blockedDoubleArray_Skel_h
/**
 * builtin method
 */
void pgas_blockedDoubleArray__ctor_impl( void* _this, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDoubleArray__ctor2_impl( void* _this, void* private_data, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDoubleArray__dtor_impl( void* _this, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDoubleArray__load_impl( void* _this, struct sidl_BaseInterface__object** _ex);
void pgas_blockedDoubleArray_allocate_impl( void* _this, int size, struct 
  sidl_BaseInterface__object** _ex);
double pgas_blockedDoubleArray_get_impl( void* _this, int idx, struct 
  sidl_BaseInterface__object** _ex);
void pgas_blockedDoubleArray_set_impl( void* _this, int idx, double val, struct 
  sidl_BaseInterface__object** _ex);
void ctor();
void dtor();
#endif

