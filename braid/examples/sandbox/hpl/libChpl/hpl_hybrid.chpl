
use ArrayWrapper;
use BlockCycDist;
use hplsupport_BlockCyclicDistArray2dDouble_chplImpl;
use hplsupport_SimpleArray1dInt_chplImpl;


config const n = 8;
config const blkSize = 2;

_extern proc panelSolve(inout abData, inout pivData, 
		/* abLimits*/ in abStart1: int(32), in abEnd1: int(32), in abStart2: int(32), in abEnd2: int(32), 
		/*panel domain*/ in start1: int(32), in end1: int(32), in start2: int(32), in end2: int(32));

type indexType = int(32);
type elemType = real(64);	
	
proc dummy_calls() {  
  	
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

  var pivWrapper = new ArrayWrapper(indexType, 1, piv);
  for i in [1..n] do {
    impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(
		  pivWrapper, 
		  impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(
  				pivWrapper, i) + i,
  		  i);
  }
  
  writeln("BEFORE: Ab:"); writeln(abWrapper.wrappedArray);
  writeln("BEFORE: piv:"); writeln(pivWrapper.wrappedArray);
  
  var useChplMethod = false;
  var b = 1;
  if (useChplMethod) {
	  var abD = ab.domain;
	  var l  = abD[b.., b..#blkSize];
	  panelSolveChpl(abWrapper.wrappedArray, l, pivWrapper.wrappedArray);
  }
  else {
    panelSolve(abWrapper, pivWrapper, 
		  1, n, 1, n + 1, 
		  b, n, b, b + blkSize - 1);
  }
  
  writeln("AFTER: Ab:"); writeln(abWrapper.wrappedArray);
  writeln("AFTER: piv:"); writeln(pivWrapper.wrappedArray);
    
}

//
// do unblocked-LU decomposition within the specified panel, update the
// pivot vector accordingly
//
proc panelSolveChpl(Ab: [] elemType,
               panel: domain,
               piv: [] indexType) {

  for k in panel.dim(2) {             // iterate through the columns
	writeln(" k = ", k);
	
    var col = panel[k.., k..k];
    
    // If there are no rows below the current column return
    if col.numIndices == 0 then return;
    
    // Find the pivot, the element with the largest absolute value.
    const ( , (pivotRow, )) = maxloc reduce(abs(Ab(col)), col);
    
    writeln("  pivot row = ", pivotRow);

    // Capture the pivot value explicitly (note that result of maxloc
    // is absolute value, so it can't be used directly).
    //
    const pivotVal = Ab[pivotRow, k];
    writeln("  pivot val = ", pivotVal);
    
    // Swap the current row with the pivot row and update the pivot vector
    // to reflect that
    Ab[k..k, ..] <=> Ab[pivotRow..pivotRow, ..];
    piv[k] <=> piv[pivotRow];

    if (pivotVal == 0) then
      halt("Matrix cannot be factorized");
    
    // divide all values below and in the same col as the pivot by
    // the pivot value
    Ab[k+1.., k..k] /= pivotVal;
    
    // update all other values below the pivot
    forall (i,j) in panel[k+1.., k+1..] do
      Ab[i,j] -= Ab[i,k] * Ab[k,j];
  }
}

proc main() {
	
  dummy_calls();
    
}