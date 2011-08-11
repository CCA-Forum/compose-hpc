// -*- chpl -*- 
// This fragment will preprocessed to yield sidl.chpl during compile time.

module sidl {
  // FIXME: autogenerate this from sidl.sidl
  _extern record sidl_BaseInterface__object {};
  _extern record sidl_BaseClass__object {};
  _extern record sidl_BaseException__object {};

  // FIXME: autogenerate this from sidl.sidl
  _extern record sidl_SIDLException__object {};
  _extern record sidl_io_Serializable__object {};

  class BaseInterface {
    var self: sidl_BaseInterface__object;
  }
  
  _extern proc sidl_BaseClass__createObject(d_data: int, inout ex: sidl_BaseInterface__object): sidl_BaseClass__object;
  
  // All the static methods of class BaseClass
  module BaseClass_static {
    proc create_BaseClass( inout _babel_param_ex: BaseException): BaseClass {
      var inst = new BaseClass();
      inst.init_BaseClass(_babel_param_ex);
      return inst;
    }
      
    proc wrap_BaseClass( in obj: sidl_BaseClass__object, inout _babel_param_ex: BaseException): BaseClass {
      var inst = new BaseClass();
      inst.wrap(obj, _babel_param_ex);
      return inst;
    }
  }
  class BaseClass {
    var self_BaseClass: sidl_BaseClass__object;
    proc init_BaseClass( inout _babel_param_ex: BaseException) {
      _extern proc IS_NOT_NULL(in aRef): bool;
      _extern proc SET_TO_NULL(inout aRef);
      var ex: sidl_BaseInterface__object;
      SET_TO_NULL(ex);
      this.self_BaseClass = sidl_BaseClass__createObject(0, ex);
      _extern proc sidl_BaseClass_addRef_stub( in self: 
      sidl_BaseClass__object, inout _ex: sidl_BaseInterface__object);
      sidl_BaseClass_addRef_stub(this.self_BaseClass, ex);
      if (IS_NOT_NULL(ex)) {
  	    _babel_param_ex = new BaseException(ex);
      }
    }
    proc wrap( in obj: sidl_BaseClass__object, inout _babel_param_ex: BaseException) {
      _extern proc IS_NOT_NULL(in aRef): bool;
      _extern proc SET_TO_NULL(inout aRef);
      var ex: sidl_BaseInterface__object;
      SET_TO_NULL(ex);
      this.self_BaseClass = obj;
      _extern proc sidl_BaseClass_addRef_stub( in self: 
      sidl_BaseClass__object, inout _ex: sidl_BaseInterface__object);
      sidl_BaseClass_addRef_stub(this.self_BaseClass, ex);
      if (IS_NOT_NULL(ex)) {
        _babel_param_ex = new BaseException(ex);
      }
    }
    proc ~BaseClass() {
      var ex: sidl_BaseInterface__object;
      _extern proc sidl_BaseClass_deleteRef_stub( in self: sidl_BaseClass__object, inout _ex: sidl_BaseInterface__object);
      sidl_BaseClass_deleteRef_stub(this.self_BaseClass, ex);
      _extern proc sidl_BaseClass__dtor_stub( in self: 
      sidl_BaseClass__object, inout _ex: sidl_BaseInterface__object);
      sidl_BaseClass__dtor_stub( this.self_BaseClass, ex);
    }
  }

  /**
   * Base class for all Exceptions at Chapel end
   */
  class BaseException {

    var self: sidl_BaseInterface__object;

    proc getException() {
      return this.self;
    }
  }

# include <array.inc>

// Start: Borrowed Array related items  
# include <barray.inc>
// End: Borrowed Array related items  

// Start: Distributed Array related items  
// FIXME Enable the include for darray.inc when we can support gasnet compilation
// # include <darray.inc>
# include <array_support.inc>
  

// End: Distributed Array related items  
}
