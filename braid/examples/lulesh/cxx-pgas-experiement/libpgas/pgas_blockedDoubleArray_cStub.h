#ifndef included_pgas_blockedDoubleArray_cStub_h
#define included_pgas_blockedDoubleArray_cStub_h
#include <pgas_blockedDoubleArray_IOR.h>
#include <pgas_blockedDoubleArray_IOR.h>
#include <sidlType.h>
#include <chpl_sidl_array.h>
#include <chpltypes.h>
/**
 * 
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 * 
 */
void pgas_blockedDoubleArray_addRef_stub( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex);
/**
 * 
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 * 
 */
void pgas_blockedDoubleArray_deleteRef_stub( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex);
/**
 * 
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 * 
 */
chpl_bool pgas_blockedDoubleArray_isSame_stub( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object* iobj, struct sidl_BaseInterface__object** _ex);
/**
 * 
 * Return the meta-data about the class implementing this interface.
 * 
 */
chpl_bool pgas_blockedDoubleArray_isType_stub( struct pgas_blockedDoubleArray__object* self, const char* name, struct sidl_BaseInterface__object** _ex);
struct sidl_ClassInfo__object* pgas_blockedDoubleArray_getClassInfo_stub( 
  struct pgas_blockedDoubleArray__object* self, struct 
  sidl_BaseInterface__object** _ex);
/**
 * Implicit built-in method: _dtor
 */
void pgas_blockedDoubleArray__dtor_stub( struct pgas_blockedDoubleArray__object* self, struct sidl_BaseInterface__object** _ex);
#endif

