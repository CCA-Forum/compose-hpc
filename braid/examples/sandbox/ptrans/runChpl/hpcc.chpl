
// All the static methods of class ParallelTranspose
module ParallelTranspose_static {

    use sidl;	
    use hplsupport;
    
    _extern proc hpcc_ParallelTranspose_ptransCompute( 
        		in a: hplsupport_BlockCyclicDistArray2dDouble, 
        		in c: hplsupport_BlockCyclicDistArray2dDouble, 
        		in beta: real(64), 
        		in i: int(32), 
        		in j: int(32), 
        		inout ex: sidl_BaseInterface);
    	
    proc ptransCompute( 
    		in a: hplsupport.BlockCyclicDistArray2dDouble, 
    		in c: hplsupport.BlockCyclicDistArray2dDouble, 
    		in beta: real(64), 
    		in i: int(32), 
    		in j: int(32)) {
        var ex: sidl_BaseInterface;
        
        hpcc_ParallelTranspose_ptransCompute( a.self, c.self, beta, i, j, ex);
    }   
    
}

class ParallelTranspose {
  
}