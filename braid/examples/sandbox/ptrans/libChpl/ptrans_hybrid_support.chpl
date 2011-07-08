use BlockCycDist;
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
  
  // Access the array from the current locale
  var curVal = impl_hplsupport_BlockCyclicDistArray2dDouble_get_chpl(ab, 2, 2);
  impl_hplsupport_BlockCyclicDistArray2dDouble_set_chpl(ab, curVal + 125.0, 2, 2);
  // Access the array from multiple locales
  for (i, j) in matVectSpace do on Locales(ab(i, j).locale.id) do {
	var newVal = impl_hplsupport_BlockCyclicDistArray2dDouble_get_chpl(
			ab, i, j) ;
	if (i < j) {
	  newVal += 10 * i + j;	
	} else {
	  newVal += i + 10 * j;
	}
	impl_hplsupport_BlockCyclicDistArray2dDouble_set_chpl(
			ab, newVal, i, j);  
  }

  writeln("dummy_calls() ends.");
}