use Overload;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

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
  else
    r = ResultType.FAIL;
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

proc test_isSliceWorking() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_isSliceWorking", sidl_ex);
  init_part(); run_part(" isSliceWorking(true)", true);
  tracker.writeComment("End: test_isSliceWorking", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_isSliceWorking();

tracker.close(sidl_ex);
