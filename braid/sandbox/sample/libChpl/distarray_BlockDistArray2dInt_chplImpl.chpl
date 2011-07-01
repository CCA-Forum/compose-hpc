
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

// End: Methods that will reside in the Skeleton file

///////////////////////////////////////////

proc main_dummy_calls() {
  writeln(" distarray_BlockDistArray2dInt_chplImpl.main_dummy_calls() starts...");
  var distArray = impl_distarray_BlockDistArray2dInt_initArray_chpl(1, 4, 1, 6, 2, 3);
  impl_distarray_BlockDistArray2dInt_setIntoArray_chpl(distArray, 42, 2, 2);
  var res = impl_distarray_BlockDistArray2dInt_getFromArray_chpl(distArray, 2, 2);
  writeln(" distarray_BlockDistArray2dInt_chplImpl.main_dummy_calls() ends.");	
}

///////////////////////////////////////////


