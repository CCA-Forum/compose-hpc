#ifndef included_pgas_blockedDouble3dArray_Skel_h
#define included_pgas_blockedDouble3dArray_Skel_h
/**
 * builtin method
 */
void pgas_blockedDouble3dArray__ctor_impl( void* _this, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDouble3dArray__ctor2_impl( void* _this, void* private_data, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDouble3dArray__dtor_impl( void* _this, struct sidl_BaseInterface__object** _ex);
/**
 * builtin method
 */
void pgas_blockedDouble3dArray__load_impl( void* _this, struct sidl_BaseInterface__object** _ex);
/**
 * 
 * allocate a blocked cubic array of doubles in size*size*size
 * 
 */
void pgas_blockedDouble3dArray_allocate_impl( void* _this, int size, struct sidl_BaseInterface__object** _ex);
double pgas_blockedDouble3dArray_get_impl( void* _this, int idx1, int idx2, int 
  idx3, struct sidl_BaseInterface__object** _ex);
void pgas_blockedDouble3dArray_set_impl( void* _this, int idx1, int idx2, int 
  idx3, double val, struct sidl_BaseInterface__object** _ex);
void ctor();
void dtor();
#endif

