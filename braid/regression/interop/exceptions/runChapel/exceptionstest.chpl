use exceptions;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

var part_no: int = 0;
var sidl_ex: BaseException = nil;
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
  else
    r = ResultType.FAIL;
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("End of part " + part_no, sidl_ex);
}

proc assertTrue(actual: bool, msg: string): bool
{
  var res = (actual == true);
  if (!res) {
    tracker.writeComment("Expected: true, Found: " + actual + ": " + msg, sidl_ex);
  }
  return res;
}

{
  // Dummy
  tracker.writeComment("Start: Exceptions test", sidl_ex);
  
  var obj: exceptions.Fib = new exceptions.Fib(sidl_ex);

  init_part(); run_part( "dummy test", true == true );

  delete obj;
  tracker.writeComment("End: Exceptions test", sidl_ex);
}

tracker.close(sidl_ex);
