
use DistArray;
use BlockDist;

// Start: Methods that will reside in the Skeleton file

proc impl_distarray_BlockDistArray2dInt_initArray_chpl(
      lo1: int, hi1: int, 
      lo2: int, hi2: int, 
      blk1: int, blk2: int) {
  var myBlockedDomLiteral = [lo1..hi1, lo2..hi2] dmapped Block([1..blk1, 1..blk2]);
  var myArray: [myBlockedDomLiteral] int(32);
  var distArray = new DistArray(myArray.eltType, myArray.rank, myArray);
  return distArray;
}

proc impl_distarray_BlockDistArray2dInt_getFromArray_chpl(distArray, idx1: int(32), idx2: int(32)) {
  return distArray.get(idx1, idx2);
}

proc impl_distarray_BlockDistArray2dInt_setIntoArray_chpl(distArray, newVal: int(32), idx1: int(32), idx2: int(32)) {
  distArray.set(newVal, idx1, idx2);
}

/**
 * C = A * B
 * using Cannon's algorithm for matrix multiplication
 * A, B and C must be square matrices.
 * There must be same number of blocks in each dimension
 */
//*/
proc impl_distarray_BlockDistArray2dInt_multiply_cannon_chpl(
		inout A, inout B, inout C,
		lo1: int, hi1: int, 
		lo2: int, hi2: int, 
		blk1: int, blk2: int) {
  // N = numblocks in either dimension, i.e. (hi - lo + 1) / blk is same in each dimesnion
  var N1 = (hi1 - lo1 + 1) / blk1;
  var N2 = (hi2 - lo2 + 1) / blk2;
  if (N1 != N2) {
	halt("The number of blocks in each dimension do not match: ", N1, " and ", N2, " blocks in each dim.");
  }
  
  // writeln("A.distArr: "); writeln(A.distArr);
  // writeln("B.distArr: "); writeln(B.distArr);  
  
  // Initial alignment: shift block-column i of B UP by i block-units, block-row i of A LEFT by i block-units
  for i in [1..(N1 - 1)] do {
	var loopLowA = lo1 + (i * blk1);
	var loopLowB = lo2 + (i * blk2);
	// writeln("   i = ", i, ", loopLowA = ", loopLowA, ", loopLowB = ", loopLowB);
	distarray_BlockDistArray2dInt_shiftLeft(A, loopLowA, loopLowA + blk1 - 1, lo2, hi2, blk1, blk2);
	distarray_BlockDistArray2dInt_shiftUp(B, lo1, hi1, loopLowB, loopLowB + blk2 - 1, blk1, blk2);
  }

  // loop N times
  for i in [1 .. N1] do {
	// TODO Need to parallelize this
	//   perform local multiplication to store items in C
    for (b1, b2) in [0.. #N1, 0.. #N2] do {
      on Locales((b1 * N1 + b2) % numLocales) do {	
        var loopLo1 = lo1 + (b1 * blk1);
        var loopLo2 = lo2 + (b2 * blk2);
        var loopHi1 = loopLo1 + blk1 - 1;
        var loopHi2 = loopLo2 + blk2 - 1;
      
        for (i1, j1) in [loopLo1..loopHi1, loopLo2..loopHi2] do
    	  for (k1, k2) in (loopLo1..loopHi1, loopLo2..loopHi2) do // zipper iteration
    	    C.distArr(i1, j1) += A.distArr(i1, k2) * B.distArr(k1, j1);
      }
    }
    if (i != N1) {
	  //   shift entire A LEFT by one block-unit
      distarray_BlockDistArray2dInt_shiftLeft(A, lo1, hi1, lo2, hi2, blk1, blk2);
      //   shift entire B UP by one block-unit
      distarray_BlockDistArray2dInt_shiftUp(B, lo1, hi1, lo2, hi2, blk1, blk2);
    }
  }
	
  // writeln("C.distArr: "); writeln(C.distArr);
  // C stores the result	
}
//**/

proc distarray_BlockDistArray2dInt_shiftLeft(inout anArray, 
		lo1: int, hi1: int, 
		lo2: int, hi2: int, 
		blk1: int, blk2: int) {
  
  var firstBlkDom: domain(2) = [lo1 .. hi1, lo2 .. #blk2];
  var firstBlkData: [firstBlkDom] int;
  
  firstBlkData[firstBlkDom] = anArray.distArr[firstBlkDom];
  
  var srcBlkDom: domain(2) = [lo1 .. hi1, (lo2 + blk2) .. hi2];
  var destBlkDom: domain(2) = [lo1 .. hi1, lo2 .. (hi2 - blk2)];
  var distArrayAlias: [destBlkDom] => anArray.distArr[srcBlkDom];
  
  var temp1BlkData: [destBlkDom] int; // Is temp necessary?
  temp1BlkData[destBlkDom] = distArrayAlias[destBlkDom];
  anArray.distArr[destBlkDom] = temp1BlkData[destBlkDom];
  
  var lastBlkDom: domain(2) = [lo1 .. hi1, (hi2 + 1 - blk2) .. #blk2];
  var distArrayAlias2: [firstBlkDom] => anArray.distArr[lastBlkDom];
  distArrayAlias2[firstBlkDom] = firstBlkData[firstBlkDom]; 
}

proc distarray_BlockDistArray2dInt_shiftUp(inout anArray, 
		lo1: int, hi1: int, 
		lo2: int, hi2: int, 
		blk1: int, blk2: int) {
  
  var firstBlkDom: domain(2) = [lo1 .. #blk1, lo2 .. hi2];
  var firstBlkData: [firstBlkDom] int;
  
  firstBlkData[firstBlkDom] = anArray.distArr[firstBlkDom];
  
  var srcBlkDom: domain(2) = [(lo1 + blk1) .. hi1, lo2 .. hi2];
  var destBlkDom: domain(2) = [lo1 .. (hi1 - blk1), lo2 .. hi2];
  var distArrayAlias: [destBlkDom] => anArray.distArr[srcBlkDom];
  
  var temp1BlkData: [destBlkDom] int; // Is temp necessary?
  temp1BlkData[destBlkDom] = distArrayAlias[destBlkDom];
  anArray.distArr[destBlkDom] = temp1BlkData[destBlkDom];
  
  var lastBlkDom: domain(2) = [(hi1 + 1 - blk1) .. #blk1, lo2 .. hi2];
  var distArrayAlias2: [firstBlkDom] => anArray.distArr[lastBlkDom];
  distArrayAlias2[firstBlkDom] = firstBlkData[firstBlkDom]; 
}

// End: Methods that will reside in the Skeleton file

///////////////////////////////////////////

proc main_dummy_calls() {
  writeln(" distarray_BlockDistArray2dInt_chplImpl.main_dummy_calls() starts...");
  var distArray = impl_distarray_BlockDistArray2dInt_initArray_chpl(1, 4, 1, 6, 2, 3);
  impl_distarray_BlockDistArray2dInt_setIntoArray_chpl(distArray, 42, 2, 2);
  var res = impl_distarray_BlockDistArray2dInt_getFromArray_chpl(distArray, 2, 2);
  
  for (i,j) in [1..4, 1..6] do 
	distArray.distArr(i, j) = 10 * i + j;  
  // writeln("Before Shift-Left:"); writeln(distArray.distArr);
  distarray_BlockDistArray2dInt_shiftLeft(distArray, 1, 4, 1, 6, 2, 3);
  // writeln("After Shift-Left:"); writeln(distArray.distArr);
  
  for (i,j) in [1..4, 1..6] do 
  	distArray.distArr(i, j) = 10 * i + j;  
  // writeln("Before Shift-Up:"); writeln(distArray.distArr);
  distarray_BlockDistArray2dInt_shiftUp(distArray, 1, 4, 1, 6, 2, 3);
  // writeln("After Shift-Up:"); writeln(distArray.distArr);
    
  var A = impl_distarray_BlockDistArray2dInt_initArray_chpl(1, 4, 1, 4, 2, 2);
  for (i,j) in [1..4, 1..4] do A.distArr(i, j) = 10 * i + j; 
  var B = impl_distarray_BlockDistArray2dInt_initArray_chpl(1, 4, 1, 4, 2, 2);
  for (i,j) in [1..4, 1..4] do B.distArr(i, j) = 10 * j + i; 
  var C = impl_distarray_BlockDistArray2dInt_initArray_chpl(1, 4, 1, 4, 2, 2);
  C.distArr = 0;
  
  writeln("A: "); writeln(A.distArr);
  writeln("B: "); writeln(B.distArr);
  impl_distarray_BlockDistArray2dInt_multiply_cannon_chpl(A, B, C, 1, 4, 1, 4, 2, 2);
  writeln("C (A * B): "); writeln(C.distArr);
  
  writeln(" distarray_BlockDistArray2dInt_chplImpl.main_dummy_calls() ends.");	
}

///////////////////////////////////////////



