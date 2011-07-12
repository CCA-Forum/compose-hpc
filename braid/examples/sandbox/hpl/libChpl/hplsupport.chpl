

use sidl;

// FIXME Avoid __object in extern record name
_extern record hplsupport_SimpleArray1dInt {};

_extern proc hplsupport_SimpleArray1dInt__createObject(
		d_data: int, 
		inout ex: sidl_BaseInterface): hplsupport_SimpleArray1dInt;

_extern proc hplsupport_SimpleArray1dInt_initData_stub( 
   		in self: hplsupport_SimpleArray1dInt, 
   		in data: int(64), 
   		inout ex: sidl_BaseInterface);

_extern proc hplsupport_SimpleArray1dInt_get_stub( 
		in self: hplsupport_SimpleArray1dInt, 
		in idx1: int(32), 
		inout ex: sidl_BaseInterface): int(32);
		
_extern proc hplsupport_SimpleArray1dInt_set_stub( 
		in self: hplsupport_SimpleArray1dInt, 
		in newVal: int(32), 
		in idx1: int(32), 
        inout ex: sidl_BaseInterface);
    
// All the static methods of class SimpleArray1dInt
module SimpleArray1dInt_static {

}

class SimpleArray1dInt {
    var self: hplsupport_SimpleArray1dInt;

    /**
     * Constructor
     */
    proc SimpleArray1dInt() {
          var ex: sidl_BaseInterface;
          this.self = hplsupport_SimpleArray1dInt__createObject(0, ex);
    }
    
    /**
     * Constructor for wrapping an existing object
     */
    proc SimpleArray1dInt( in obj: hplsupport_SimpleArray1dInt) {
          this.self = obj;
    }
    
    proc initData( in data: int(64)) {
        var ex:sidl_BaseInterface;
        hplsupport_SimpleArray1dInt_initData_stub( self, data, ex);
    }
    
    proc get( in idx1: int(32)): int(32) {
        var ex:sidl_BaseInterface;
        return hplsupport_SimpleArray1dInt_get_stub( self, idx1, ex);
    }
    
    proc set( in newVal: int(32), in idx1: int(32)) {
        var ex:sidl_BaseInterface;
        hplsupport_SimpleArray1dInt_set_stub( self, newVal, idx1, ex);
    }
    
    
};


use sidl;

// FIXME Avoid __object in extern record name
_extern record hplsupport_BlockCyclicDistArray2dDouble {};

_extern proc hplsupport_BlockCyclicDistArray2dDouble__createObject(
		d_data: int, 
		inout ex: sidl_BaseInterface): hplsupport_BlockCyclicDistArray2dDouble;

_extern proc hplsupport_BlockCyclicDistArray2dDouble_initData_stub( 
		in self: hplsupport_BlockCyclicDistArray2dDouble, 
		in data: int(64), 
		inout ex: sidl_BaseInterface);

_extern proc hplsupport_BlockCyclicDistArray2dDouble_get_stub( 
		in self: hplsupport_BlockCyclicDistArray2dDouble, 
		in idx1: int(32), 
		in idx2: int(32), 
		inout ex: sidl_BaseInterface): real(64);
		
_extern proc hplsupport_BlockCyclicDistArray2dDouble_set_stub( 
		in self: hplsupport_BlockCyclicDistArray2dDouble, 
		in newVal: real(64), 
		in idx1: int(32), 
		in idx2: int(32), 
		inout ex: sidl_BaseInterface);
    
// All the static methods of class BlockCyclicDistArray2dDouble
module BlockCyclicDistArray2dDouble_static {

}

class BlockCyclicDistArray2dDouble {
    var self: hplsupport_BlockCyclicDistArray2dDouble;
    
    /**
     * Constructor
     */
    proc BlockCyclicDistArray2dDouble() {
          var ex: sidl_BaseInterface;
          this.self = hplsupport_BlockCyclicDistArray2dDouble__createObject(0, ex);
    }
    
    /**
     * Constructor for wrapping an existing object
     */
    proc BlockCyclicDistArray2dDouble( in obj: hplsupport_BlockCyclicDistArray2dDouble) {
          this.self = obj;
    }
        
    proc initData( in data: int(64)) {
        var ex:sidl_BaseInterface;
        hplsupport_BlockCyclicDistArray2dDouble_initData_stub( self, data, ex);
    }
    
    proc get( in idx1: int(32), in idx2: int(32)): real(64) {
        var ex:sidl_BaseInterface;
        return hplsupport_BlockCyclicDistArray2dDouble_get_stub( self, idx1, idx2, ex);
    }
    
    proc set( in newVal: real(64), in idx1: int(32), in idx2: int(32)) {
        var ex:sidl_BaseInterface;
        hplsupport_BlockCyclicDistArray2dDouble_set_stub( self, newVal, idx1, idx2, ex);
    }
    
}; 
