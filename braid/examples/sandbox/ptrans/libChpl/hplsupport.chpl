

use sidl;

/**
 * FIXME We refer to int(32) for a 32b machine and int(64) for a 64b machine
 */

_extern record hplsupport_BlockCyclicDistArray2dDouble {};
_extern proc hplsupport_BlockCyclicDistArray2dDouble__createObject(d_data: int, 
  inout ex: sidl_BaseInterface): 
  hplsupport_BlockCyclicDistArray2dDouble;

    /**
     * FIXME: The generator should set the type of data to opaque instead of int(32)
     */
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_initData_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble, in data: opaque, inout ex: 
        sidl_BaseInterface);
    
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_get_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble, in idx1: int(32), in idx2: 
        int(32), inout ex: sidl_BaseInterface): real(64);
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_set_stub( in self: 
        hplsupport_BlockCyclicDistArray2dDouble, in newVal: real(64), in idx1: 
        int(32), in idx2: int(32), inout ex: sidl_BaseInterface);
    
    
// All the static methods of class BlockCyclicDistArray2dDouble
module BlockCyclicDistArray2dDouble_static {

    use sidl;	
    
    _extern proc hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( 
        		in a: hplsupport_BlockCyclicDistArray2dDouble, 
        		in c: hplsupport_BlockCyclicDistArray2dDouble, 
        		in beta: real(64), 
        		in i: int(32), 
        		in j: int(32), 
        		inout ex: sidl_BaseInterface);
    	
    /**
     * FIXME C is inout, the generator should use GET_REF to obtain a reference?
     */	
    proc ptransHelper( 
    		in a: hplsupport.BlockCyclicDistArray2dDouble, 
    		in c: hplsupport.BlockCyclicDistArray2dDouble, 
    		in beta: real(64), 
    		in i: int(32), 
    		in j: int(32)) {
        var ex: sidl_BaseInterface;
        
        hplsupport_BlockCyclicDistArray2dDouble_ptransHelper_stub( a.self, c.self, 
        		beta, i, j, ex);
        // TODO Figure out how to handle the inout result
    }
    
    
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
    
    /**
     * FIXME We refer to int(32) for a 32b machine and int(64) for a 64b machine?
     * FIXME The generator should use a macro (e.g. GET_REF()) to pass the address of data to the stub
     */
    proc initData( in data) {
    	_extern proc GET_REF(inData): opaque;
    	
        var ex:sidl_BaseInterface;
        hplsupport_BlockCyclicDistArray2dDouble_initData_stub( self, GET_REF(data), ex);
    }
    
    proc get( in idx1: int(32), in idx2: int(32)): real(64) {
        var ex:sidl_BaseInterface;
        return hplsupport_BlockCyclicDistArray2dDouble_get_stub( self, idx1, idx2, ex);
    }
    
    proc set( in newVal: real(64), in idx1: int(32), in idx2: int(32)) {
        var ex:sidl_BaseInterface;
        hplsupport_BlockCyclicDistArray2dDouble_set_stub( self, newVal, idx1, idx2, 
            ex);
    }
    
    
};
;
