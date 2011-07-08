

use sidl;

/**
 * FIXME We refer to int(32) for a 32b machine and int(64) for a 64b machine
 */
_extern proc GET_REF(inData: int(32)): opaque;

_extern record hplsupport_BlockCyclicDistArray2dDouble__object {};
_extern proc hplsupport_BlockCyclicDistArray2dDouble__createObject(d_data: int, 
  inout ex: sidl_BaseInterface__object): 
  hplsupport_BlockCyclicDistArray2dDouble__object;

    /**
     * FIXME: The generator should set the type of data to opaque instead of int(32)
     */
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_initData_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble__object, in data: opaque, inout ex: 
        sidl_BaseInterface__object);
    
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_get_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble__object, in idx1: int(32), in idx2: 
        int(32), inout ex: sidl_BaseInterface__object): real(64);
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_set_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble__object, in newVal: real(64), in idx1: 
        int(32), in idx2: int(32), inout ex: sidl_BaseInterface__object);
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( in a: 
        hplsupport_BlockCyclicDistArray2dDouble__object, inout c: 
        hplsupport_BlockCyclicDistArray2dDouble__object, in beta: real(64), in i: 
        int(32), in j: int(32), inout ex: sidl_BaseInterface__object);
    
// All the static methods of class BlockCyclicDistArray2dDouble
module BlockCyclicDistArray2dDouble_static {

    proc ptransHelper( in a: hplsupport.BlockCyclicDistArray2dDouble, inout c: hplsupport.BlockCyclicDistArray2dDouble, in beta: real(64), in i: int(32), in j: int(32)) {
        var ex:sidl_BaseInterface__object;
        var _IOR_c:hplsupport_BlockCyclicDistArray2dDouble__object;
        hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( a.ior, _IOR_c, beta, 
            i, j, ex);
        c = new hplsupport.BlockCyclicDistArray2dDouble( _IOR_c);
    }
    
    
}
class BlockCyclicDistArray2dDouble {
var self: hplsupport_BlockCyclicDistArray2dDouble__object;
    /**
     * Constructor
     */
    proc BlockCyclicDistArray2dDouble() {
          var ex: sidl_BaseInterface__object;
          this.self = hplsupport_BlockCyclicDistArray2dDouble__createObject(0, ex);
    }
    
    /**
     * Constructor for wrapping an existing object
     */
    proc BlockCyclicDistArray2dDouble( in obj: hplsupport_BlockCyclicDistArray2dDouble__object) {
          this.self = obj;
    }
    
    /**
     * FIXME We refer to int(32) for a 32b machine and int(64) for a 64b machine
     * FIXME The generator should use a macro (e.g. GET_REF()) to pass the address of data to the stub
     * FIXME Can we make the generator not type the parameter data in this method (i.e. a generic method)?
     */
    proc initData( in data) {
        var ex:sidl_BaseInterface__object;
        hplsupport_BlockCyclicDistArray2dDouble_initData_stub( self, GET_REF(data), ex);
    }
    
    proc get( in idx1: int(32), in idx2: int(32)): real(64) {
        var ex:sidl_BaseInterface__object;
        return hplsupport_BlockCyclicDistArray2dDouble_get_stub( self, idx1, idx2, ex);
    }
    
    proc set( in newVal: real(64), in idx1: int(32), in idx2: int(32)) {
        var ex:sidl_BaseInterface__object;
        hplsupport_BlockCyclicDistArray2dDouble_set_stub( self, newVal, idx1, idx2, 
            ex);
    }
    
    
};
;
