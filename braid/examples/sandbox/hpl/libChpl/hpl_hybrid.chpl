
use ArrayWrapper;
use BlockCycDist;
use hplsupport_BlockCyclicDistArray2dDouble_chplImpl;
use hplsupport_SimpleArray1dInt_chplImpl;


config const n = 8;
config const blkSize = 2;

proc dummy_calls() {
  
  type indexType = int(32);
  type elemType = real(64);	
		
  const matVectSpace: domain(2, indexType) dmapped BlockCyclic(startIdx=(1, 1), (blkSize, blkSize)) = [1..n, 1..n+1];	
  var ab : [matVectSpace] elemType;  // the matrix A and vector b
  var piv: [1..n] indexType;         // a vector of pivot values
  
  var abWrapper = new ArrayWrapper(elemType, 2, ab);
  impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
		  abWrapper, 
		  impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(
				  abWrapper, 2, 2) + 125.0,
		  2, 2);

  var pivWrapper = new ArrayWrapper(indexType, 1, piv);
  impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(
		  pivWrapper, 
		  impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(
  				pivWrapper, 2) + 100,
  		  2);
  
  writeln("Ab:"); writeln(abWrapper.wrappedArray);
  writeln("piv:"); writeln(pivWrapper.wrappedArray);	

}

proc main() {
	
  dummy_calls();
    
}