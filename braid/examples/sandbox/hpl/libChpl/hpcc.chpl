
// All the static methods of class HighPerformanceLinpack
module HighPerformanceLinpack_static {

    use sidl;	
    use hplsupport;
    
    _extern proc hpcc_HighPerformanceLinpack_panelSolveCompute( 
        		in ab: hplsupport_BlockCyclicDistArray2dDouble, 
        		in piv: hplsupport_SimpleArray1dInt, 
        		/* abLimits*/ 
        		in abStart1: int(32), 
        		in abEnd1: int(32), 
        		in abStart2: int(32), 
        		in abEnd2: int(32), 
        		/*panel domain*/ 
        		in start1: int(32), 
        		in end1: int(32), 
        		in start2: int(32), 
        		in end2: int(32), 
        		inout ex: sidl_BaseInterface);
    	
    proc panelSolveCompute( 
    		in ab: hplsupport.BlockCyclicDistArray2dDouble, 
    		in piv: hplsupport.SimpleArray1dInt, 
    		/* abLimits*/ 
    		in abStart1: int(32), 
    		in abEnd1: int(32), 
    		in abStart2: int(32), 
    		in abEnd2: int(32), 
    		/*panel domain*/ 
    		in start1: int(32), 
    		in end1: int(32), 
    		in start2: int(32), 
    		in end2: int(32)) {
        var ex: sidl_BaseInterface;
        
        hpcc_HighPerformanceLinpack_panelSolveCompute( ab.self, piv.self, 
        		abStart1, abEnd1, abStart2, abEnd2, 
        		start1, end1, start2, end2, 
        		ex);
    }   
    
}

class HighPerformanceLinpack {
  
}