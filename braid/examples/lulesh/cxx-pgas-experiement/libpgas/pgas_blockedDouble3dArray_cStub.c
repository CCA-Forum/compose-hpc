#include <pgas.h>
#include <pgas_blockedDouble3dArray_cStub.h>
/**
 * 
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 * 
 */
void pgas_blockedDouble3dArray_addRef_stub( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  (*self->d_epv->f_addRef)( self,   _ex);
}

/**
 * 
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 * 
 */
void pgas_blockedDouble3dArray_deleteRef_stub( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  (*self->d_epv->f_deleteRef)( self,   _ex);
}

/**
 * 
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 * 
 */
chpl_bool pgas_blockedDouble3dArray_isSame_stub( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object* iobj, struct sidl_BaseInterface__object** _ex) {
sidl_bool _ior__retval;
  chpl_bool _retval;
  _ior__retval = (*self->d_epv->f_isSame)( self,   iobj,   _ex);
  /* sidl_bool is an int, but chapel bool is a char/_Bool */
  _retval = (chpl_bool)_ior__retval;
  return _retval;
}

/**
 * 
 * Return the meta-data about the class implementing this interface.
 * 
 */
chpl_bool pgas_blockedDouble3dArray_isType_stub( struct pgas_blockedDouble3dArray__object* self, const char* name, struct sidl_BaseInterface__object** _ex) {
const char* _ior_name;
  sidl_bool _ior__retval;
  chpl_bool _retval;
  _ior_name = name;
  _ior__retval = (*self->d_epv->f_isType)( self,   _ior_name,   _ex);
  /* sidl_bool is an int, but chapel bool is a char/_Bool */
  _retval = (chpl_bool)_ior__retval;
  return _retval;
}

struct sidl_ClassInfo__object* pgas_blockedDouble3dArray_getClassInfo_stub( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
struct sidl_ClassInfo__object* _retval;
  _retval = (*self->d_epv->f_getClassInfo)( self,   _ex);
  return _retval;
}

/**
 * Implicit built-in method: _dtor
 */
void pgas_blockedDouble3dArray__dtor_stub( struct pgas_blockedDouble3dArray__object* self, struct sidl_BaseInterface__object** _ex) {
  (*self->d_epv->f__dtor)( self,   _ex);
}
