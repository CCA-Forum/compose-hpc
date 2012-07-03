

use sidl;
use pgas_Impl;
extern record pgas_MPIinitializer__object {
};
extern proc _cast_sidl_BaseClass(in ior: pgas_MPIinitializer__object, out ex: sidl_BaseInterface__object): sidl.sidl_BaseClass__object;
extern proc sidl_BaseClass_cast_pgas_MPIinitializer(in ior: sidl.sidl_BaseClass__object): pgas_MPIinitializer__object;
extern proc _cast_sidl_BaseInterface(in ior: pgas_MPIinitializer__object, out ex: sidl_BaseInterface__object): sidl.sidl_BaseInterface__object;
extern proc sidl_BaseInterface_cast_pgas_MPIinitializer(in ior: sidl.sidl_BaseInterface__object): pgas_MPIinitializer__object;
extern proc pgas_MPIinitializer__createObject(d_data: int, out ex: sidl_BaseInterface__object): pgas_MPIinitializer__object;
// All the static methods of class MPIinitializer
module MPIinitializer_static {


    /**
     * Static helper function to create instance using create()
     */
    proc create(out _babel_param_ex: BaseInterface): MPIinitializer {
          var inst = new MPIinitializer();
          inst.init_MPIinitializer(_babel_param_ex);
          return inst;
    }
    
    
    /**
     * Static helper function to create instance using wrap()
     */
    proc wrap_MPIinitializer(in obj: pgas_MPIinitializer__object,     out _babel_param_ex: BaseInterface): MPIinitializer {
          var inst = new MPIinitializer();
          inst.wrap(obj, _babel_param_ex);
          return inst;
    }
    
}

class MPIinitializer  /*: sidl*/   {
var self_MPIinitializer: pgas_MPIinitializer__object;
    
    /**
     * Pseudo-Constructor to initialize the IOR object
     */
    proc init_MPIinitializer(out _babel_param_ex: BaseInterface) {
        extern proc pgas_MPIinitializer_addRef_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
          extern proc IS_NOT_NULL(in aRef): bool;
          extern proc SET_TO_NULL(inout aRef);
          var ex: sidl_BaseInterface__object;
          SET_TO_NULL(ex);
          this.self_MPIinitializer = pgas_MPIinitializer__createObject(0, ex);
        pgas_MPIinitializer_addRef_stub(this.self_MPIinitializer,         ex);
          if (IS_NOT_NULL(ex)) {
             _babel_param_ex = new BaseInterface(ex);
          }
    }
    
    
    /**
     * Pseudo-Constructor for wrapping an existing object
     */
    proc wrap(in obj: pgas_MPIinitializer__object,     out _babel_param_ex: BaseInterface) {
        extern proc pgas_MPIinitializer_addRef_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
          extern proc IS_NOT_NULL(in aRef): bool;
          extern proc SET_TO_NULL(inout aRef);
          var ex: sidl_BaseInterface__object;
          SET_TO_NULL(ex);
          this.self_MPIinitializer = obj;
        pgas_MPIinitializer_addRef_stub(this.self_MPIinitializer,         ex);
          if (IS_NOT_NULL(ex)) {
             _babel_param_ex = new BaseInterface(ex);
          }
    }
    
    
    /**
     * Destructor
     */
    proc ~MPIinitializer() {
        extern proc pgas_MPIinitializer_deleteRef_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        extern proc pgas_MPIinitializer__dtor_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        var ex: sidl_BaseInterface__object;
        pgas_MPIinitializer_deleteRef_stub(this.self_MPIinitializer,         ex);
        pgas_MPIinitializer__dtor_stub(this.self_MPIinitializer,         ex);
    }
    
    
    /**
     * return the current IOR pointer
     */
    proc as_pgas_MPIinitializer(): pgas_MPIinitializer__object {
        return self_MPIinitializer;
    }
    
    
    /**
     * Create a up-casted version of the IOR pointer for
     * use with the alternate constructor
     */
    proc as_sidl_BaseClass(): sidl.sidl_BaseClass__object {
        var ex: sidl_BaseInterface__object;
        return _cast_sidl_BaseClass(this.self_MPIinitializer, ex);
    }
    
    
    /**
     * Create a up-casted version of the IOR pointer for
     * use with the alternate constructor
     */
    proc as_sidl_BaseInterface(): sidl.sidl_BaseInterface__object {
        var ex: sidl_BaseInterface__object;
        return _cast_sidl_BaseInterface(this.self_MPIinitializer, ex);
    }
    

    
    /**
     * 
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in <code>sidl</code> have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     * 
     */
    proc addRef(out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        extern proc pgas_MPIinitializer_addRef_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_MPIinitializer_addRef_stub(this.self_MPIinitializer,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
    
    /**
     * 
     * Return true if and only if <code>obj</code> refers to the same
     * object as this object.
     * 
     */
    proc deleteRef(out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        extern proc pgas_MPIinitializer_deleteRef_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_MPIinitializer_deleteRef_stub(this.self_MPIinitializer,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
    
    /**
     * 
     * Return whether this object is an instance of the specified type.
     * The string name must be the <code>sidl</code> type name.  This
     * routine will return <code>true</code> if and only if a cast to
     * the string type name would succeed.
     * 
     */
    proc isSame(in iobj: sidl.BaseInterface,     out _babel_param_ex: BaseInterface): bool {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: bool;
        extern proc pgas_MPIinitializer_isSame_stub(in self: 
            pgas.pgas_MPIinitializer__object,     inout iobj: 
            sidl.sidl_BaseInterface__object,     out _ex: sidl.sidl_BaseInterface__object): 
            bool;;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_MPIinitializer_isSame_stub(this.self_MPIinitializer,         iobj.self_BaseInterface,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        return _IOR__retval;
    }
    
    
    /**
     * 
     * Return the meta-data about the class implementing this interface.
     * 
     */
    proc isType(in name: string,     out _babel_param_ex: BaseInterface): bool {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: bool;
        extern proc pgas_MPIinitializer_isType_stub(in self: 
            pgas.pgas_MPIinitializer__object,     in name: string,     out _ex: 
            sidl.sidl_BaseInterface__object): bool;;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_MPIinitializer_isType_stub(this.self_MPIinitializer,         name,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        return _IOR__retval;
    }
    
    
    proc getClassInfo(out _babel_param_ex: BaseInterface): sidl.ClassInfo {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: sidl.sidl_ClassInfo__object;
        extern proc pgas_MPIinitializer_getClassInfo_stub(in self: 
            pgas.pgas_MPIinitializer__object,     out _ex: 
            sidl.sidl_BaseInterface__object): sidl.sidl_ClassInfo__object;;
        var _retval: sidl.ClassInfo;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_MPIinitializer_getClassInfo_stub(this.self_MPIinitializer,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        _retval = sidl.ClassInfo_static.wrap_ClassInfo(_IOR__retval,         _babel_param_ex);
        return _retval;
    }
    
    
    /**
     * 
     * let's see if initializing Chapels MPI spawner first improves the situation
     * 
     */
    proc init(out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_Impl.init(this.self_MPIinitializer,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
}


use sidl;
use pgas_Impl;
extern record pgas_blockedDouble3dArray__object {
};
extern proc _cast_sidl_BaseClass(in ior: pgas_blockedDouble3dArray__object, out ex: sidl_BaseInterface__object): sidl.sidl_BaseClass__object;
extern proc sidl_BaseClass_cast_pgas_blockedDouble3dArray(in ior: sidl.sidl_BaseClass__object): pgas_blockedDouble3dArray__object;
extern proc _cast_sidl_BaseInterface(in ior: pgas_blockedDouble3dArray__object, out ex: sidl_BaseInterface__object): sidl.sidl_BaseInterface__object;
extern proc sidl_BaseInterface_cast_pgas_blockedDouble3dArray(in ior: sidl.sidl_BaseInterface__object): pgas_blockedDouble3dArray__object;
extern proc pgas_blockedDouble3dArray__createObject(d_data: int, out ex: sidl_BaseInterface__object): pgas_blockedDouble3dArray__object;
// All the static methods of class blockedDouble3dArray
module blockedDouble3dArray_static {


    /**
     * Static helper function to create instance using create()
     */
    proc create(out _babel_param_ex: BaseInterface): blockedDouble3dArray {
          var inst = new blockedDouble3dArray();
          inst.init_blockedDouble3dArray(_babel_param_ex);
          return inst;
    }
    
    
    /**
     * Static helper function to create instance using wrap()
     */
    proc wrap_blockedDouble3dArray(in obj: pgas_blockedDouble3dArray__object,     out _babel_param_ex: BaseInterface): blockedDouble3dArray {
          var inst = new blockedDouble3dArray();
          inst.wrap(obj, _babel_param_ex);
          return inst;
    }
    
}

class blockedDouble3dArray  /*: sidl*/   {
var self_blockedDouble3dArray: pgas_blockedDouble3dArray__object;
    
    /**
     * Pseudo-Constructor to initialize the IOR object
     */
    proc init_blockedDouble3dArray(out _babel_param_ex: BaseInterface) {
        extern proc pgas_blockedDouble3dArray_addRef_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
          extern proc IS_NOT_NULL(in aRef): bool;
          extern proc SET_TO_NULL(inout aRef);
          var ex: sidl_BaseInterface__object;
          SET_TO_NULL(ex);
          this.self_blockedDouble3dArray = pgas_blockedDouble3dArray__createObject(0, ex);
        pgas_blockedDouble3dArray_addRef_stub(this.self_blockedDouble3dArray,         ex);
          if (IS_NOT_NULL(ex)) {
             _babel_param_ex = new BaseInterface(ex);
          }
    }
    
    
    /**
     * Pseudo-Constructor for wrapping an existing object
     */
    proc wrap(in obj: pgas_blockedDouble3dArray__object,     out _babel_param_ex: BaseInterface) {
        extern proc pgas_blockedDouble3dArray_addRef_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
          extern proc IS_NOT_NULL(in aRef): bool;
          extern proc SET_TO_NULL(inout aRef);
          var ex: sidl_BaseInterface__object;
          SET_TO_NULL(ex);
          this.self_blockedDouble3dArray = obj;
        pgas_blockedDouble3dArray_addRef_stub(this.self_blockedDouble3dArray,         ex);
          if (IS_NOT_NULL(ex)) {
             _babel_param_ex = new BaseInterface(ex);
          }
    }
    
    
    /**
     * Destructor
     */
    proc ~blockedDouble3dArray() {
        extern proc pgas_blockedDouble3dArray_deleteRef_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        extern proc pgas_blockedDouble3dArray__dtor_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        var ex: sidl_BaseInterface__object;
        pgas_blockedDouble3dArray_deleteRef_stub(this.self_blockedDouble3dArray,         ex);
        pgas_blockedDouble3dArray__dtor_stub(this.self_blockedDouble3dArray,         ex);
    }
    
    
    /**
     * return the current IOR pointer
     */
    proc as_pgas_blockedDouble3dArray(): pgas_blockedDouble3dArray__object {
        return self_blockedDouble3dArray;
    }
    
    
    /**
     * Create a up-casted version of the IOR pointer for
     * use with the alternate constructor
     */
    proc as_sidl_BaseClass(): sidl.sidl_BaseClass__object {
        var ex: sidl_BaseInterface__object;
        return _cast_sidl_BaseClass(this.self_blockedDouble3dArray, ex);
    }
    
    
    /**
     * Create a up-casted version of the IOR pointer for
     * use with the alternate constructor
     */
    proc as_sidl_BaseInterface(): sidl.sidl_BaseInterface__object {
        var ex: sidl_BaseInterface__object;
        return _cast_sidl_BaseInterface(this.self_blockedDouble3dArray, ex);
    }
    

    
    /**
     * 
     * Decrease by one the intrinsic reference count in the underlying
     * object, and delete the object if the reference is non-positive.
     * Objects in <code>sidl</code> have an intrinsic reference count.
     * Clients should call this method whenever they remove a
     * reference to an object or interface.
     * 
     */
    proc addRef(out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        extern proc pgas_blockedDouble3dArray_addRef_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_blockedDouble3dArray_addRef_stub(this.self_blockedDouble3dArray,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
    
    /**
     * 
     * Return true if and only if <code>obj</code> refers to the same
     * object as this object.
     * 
     */
    proc deleteRef(out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        extern proc pgas_blockedDouble3dArray_deleteRef_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object);;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_blockedDouble3dArray_deleteRef_stub(this.self_blockedDouble3dArray,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
    
    /**
     * 
     * Return whether this object is an instance of the specified type.
     * The string name must be the <code>sidl</code> type name.  This
     * routine will return <code>true</code> if and only if a cast to
     * the string type name would succeed.
     * 
     */
    proc isSame(in iobj: sidl.BaseInterface,     out _babel_param_ex: BaseInterface): bool {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: bool;
        extern proc pgas_blockedDouble3dArray_isSame_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     inout iobj: 
            sidl.sidl_BaseInterface__object,     out _ex: sidl.sidl_BaseInterface__object): 
            bool;;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_blockedDouble3dArray_isSame_stub(this.self_blockedDouble3dArray,         iobj.self_BaseInterface,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        return _IOR__retval;
    }
    
    
    /**
     * 
     * Return the meta-data about the class implementing this interface.
     * 
     */
    proc isType(in name: string,     out _babel_param_ex: BaseInterface): bool {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: bool;
        extern proc pgas_blockedDouble3dArray_isType_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     in name: string,     out _ex: 
            sidl.sidl_BaseInterface__object): bool;;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_blockedDouble3dArray_isType_stub(this.self_blockedDouble3dArray,         name,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        return _IOR__retval;
    }
    
    
    proc getClassInfo(out _babel_param_ex: BaseInterface): sidl.ClassInfo {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: sidl.sidl_ClassInfo__object;
        extern proc pgas_blockedDouble3dArray_getClassInfo_stub(in self: 
            pgas.pgas_blockedDouble3dArray__object,     out _ex: 
            sidl.sidl_BaseInterface__object): sidl.sidl_ClassInfo__object;;
        var _retval: sidl.ClassInfo;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_blockedDouble3dArray_getClassInfo_stub(this.self_blockedDouble3dArray,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        _retval = sidl.ClassInfo_static.wrap_ClassInfo(_IOR__retval,         _babel_param_ex);
        return _retval;
    }
    
    
    /**
     * 
     * allocate a blocked cubic array of doubles in size*size*size
     * 
     */
    proc allocate(in size: int(32),     out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_Impl.allocate(this.self_blockedDouble3dArray,         size,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
    
    proc get(in idx1: int(32),     in idx2: int(32),     in idx3: int(32),     out _babel_param_ex: BaseInterface): real(64) {
        var _ex: sidl.sidl_BaseInterface__object;
        var _IOR__retval: real(64);
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        _IOR__retval = pgas_Impl.get(this.self_blockedDouble3dArray,         idx1,         idx2,         idx3,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

        return _IOR__retval;
    }
    
    
    proc set(in idx1: int(32),     in idx2: int(32),     in idx3: int(32),     in val: real(64),     out _babel_param_ex: BaseInterface) {
        var _ex: sidl.sidl_BaseInterface__object;
        
        extern proc IS_NOT_NULL(in aRef): bool;
        extern proc SET_TO_NULL(inout aRef);
        SET_TO_NULL(_ex);
        pgas_Impl.set(this.self_blockedDouble3dArray,         idx1,         idx2,         idx3,         val,         _ex);
        if (IS_NOT_NULL(_ex)) {
          _babel_param_ex = new BaseInterface( _ex);
        }

    }
    
}

