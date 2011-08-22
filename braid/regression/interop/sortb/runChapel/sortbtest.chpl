use sortb;
use synch;
use sidl;

// create a new folder sortb
// rename sort.sidl to sortb.sidl
// rename internal packages to sortb
// rename classes to QuickSortB, HeapSortB, MergeSortB
// sample commands
// >perl -pi -e 's/sort_/sortb_/g' *
// >perl -pi -e 's/sort\./sortb\./g' *
// >perl -pi -e 's/QuickSort/QuickSortB/g' *
// >rename 's/QuickSort/QuickSortB/' sort*

config var bindir = "gantlet compatibility";

var failed: bool = false;
var part_no: int = 0;
var sidl_ex: BaseException = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);

proc init_part() {
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part " + part_no, sidl_ex);
}

proc run_part(result: bool) {
  run_part("", result);
}

proc run_part(msg: string, result: bool) {
  if (msg.length > 0) {
    tracker.writeComment(msg, sidl_ex);
  }
  var r: ResultType;
  if (result) then
    r = ResultType.PASS;
  else {
    r = ResultType.FAIL;
    failed = true;
  }
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("End of part " + part_no, sidl_ex);
}

/**
 * Fill the stack with random junk.
 */
proc clearstack(magicNumber: int): int {
  return magicNumber;
}

var magicNumber = 13;
tracker.setExpectations(-1, sidl_ex);
// tracker.setExpectations(19);

proc test_Sort() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_Sort", sidl_ex);
  
  
  var merge: sortb.MergeSortB  = sortb.MergeSortB_static.create_MergeSortB(sidl_ex);
  var quick: sortb.QuickSortB  = sortb.QuickSortB_static.create_QuickSortB(sidl_ex);
  var heap:  sortb.HeapSortB  = sortb.HeapSortB_static.create_HeapSortB(sidl_ex);
  
  /*
  // FIXME: We need to support arrays of objects 
  var algs: sidl.array<sortb.SortingAlgorithm>  = sidl.array<sortb.SortingAlgorithm>.create1d(3);
  
  algs.set(0, merge);
  algs.set(1, quick);
  algs.set(2, heap);
  
  init_part(); run_part(" sortb.SortTest.stressTest", sortb.SortTest.stressTest(algs, sidl_ex));  
  */
  init_part(); run_part(" dummy", true);
  
  tracker.writeComment("End: test_Sort", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_Sort();

tracker.close(sidl_ex);

if (failed) then
  exit(1);
