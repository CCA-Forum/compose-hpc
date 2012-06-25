use sorting;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

var failed: bool = false;
var part_no: int(32) = 0;
var sidl_ex: BaseInterface = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);

proc init_part()
{
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part "+part_no, sidl_ex);
}

proc run_part(result: bool)
{
  run_part("", result);
}

proc run_part(msg: string, result: bool)
{
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
  tracker.writeComment("End of part "+part_no, sidl_ex);
}

{ 
  var magicNumber = 1;
  var merge = sorting.Mergesort_static.create(sidl_ex);
  var quick = sorting.Quicksort_static.create(sidl_ex);
  var heap = sorting.Heapsort_static.create(sidl_ex);
  //  sidl::array<sort::SortingAlgorithm> algs = sidl::array<sort::SortingAlgorithm>::create1d(3);
  tracker.setExpectations(4:int(32), sidl_ex);
//  init_part(); run_part( (merge._not_nil());
//  init_part(); run_part( (quick._not_nil());
//  init_part(); run_part( (heap._not_nil());
//  algs.set(0, merge);
//  algs.set(1, quick);
//  algs.set(2, heap);
//  init_part(); run_part( (sort.SortTest_static.stressTest(algs));  
} 
tracker.close(sidl_ex);

if (failed) then
  exit(1);
