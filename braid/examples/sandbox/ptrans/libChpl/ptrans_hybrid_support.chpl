use BlockCycDist;
use ArrayWrapper;
use hplsupport_BlockCyclicDistArray2dDouble_chplImpl;

type indexType = int(32);
type elemType = real(64);

proc ptrans_support_dummy_calls() {  
  	
  writeln("dummy_calls() starts...");
  
  var n = 8;
  var blkSize = 2;
  	
  const matVectSpace: domain(2, indexType) dmapped 
		  BlockCyclic(startIdx=(1, 1), (blkSize, blkSize)) = [1..n, 1..n+1];	
  var ab : [matVectSpace] elemType;  // the matrix A and vector b
  var piv: [1..n] indexType;         // a vector of pivot values
  
  var abWrapper = new ArrayWrapper(elemType, 2, ab);
  impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
		  abWrapper, 
		  impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(
				  abWrapper, 2, 2) + 125.0,
		  2, 2);
  for (i, j) in matVectSpace do {
	var newVal = 0;
	if (i < j) {
	  newVal = 10 * i + j;	
	} else {
	  newVal = i + 10 * j;
	}
	impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
	  		  abWrapper, newVal, i, j);  
  }

  writeln("dummy_calls() ends.");
}