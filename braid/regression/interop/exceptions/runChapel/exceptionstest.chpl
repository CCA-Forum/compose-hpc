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
  tracker.writeComment("Start: Exceptions test fib-good", sidl_ex);

  var obj: exceptions.Fib = new exceptions.Fib(sidl_ex);

  init_part();
  var baseEx: BaseException = nil;
  var isExNotNil = (baseEx != nil);
  writeln("1. isExNotNil = ", isExNotNil);
  writeln("1. baseEx = ", baseEx);
  obj.getFib(10, 25, 200, 0, baseEx);
  writeln("2. baseEx = ", baseEx);
  isExNotNil = (baseEx != nil);
  writeln("2. isExNotNil = ", isExNotNil);
  run_part( "test fib-good", baseEx == nil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-good", sidl_ex);
}

{
  tracker.writeComment("Start: Exceptions test fib-NegativeValueException", sidl_ex);

  var obj: exceptions.Fib = new exceptions.Fib(sidl_ex);

  init_part();
  var baseEx: BaseException = nil;
  obj.getFib(-1, 10, 10, 0, baseEx);
  writeln("baseEx = ", baseEx);
  var isExNotNil = (baseEx != nil);
  writeln("isExNotNil = ", isExNotNil);

  run_part( "test fib-NegativeValueException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-NegativeValueException", sidl_ex);
}

{
  tracker.writeComment("Start: Exceptions test fib-TooDeepException", sidl_ex);

  var obj: exceptions.Fib = new exceptions.Fib(sidl_ex);

  init_part();
  var baseEx: BaseException = nil;
  obj.getFib(10, 1, 1000, 0, baseEx);
  writeln("baseEx = ", baseEx);
  // FIXME Need to catch the appropriate class of exception in try-catch-like block
  var isExNotNil = (baseEx != nil);
  writeln("isExNotNil = ", isExNotNil);

  run_part( "test fib-TooDeepException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-TooDeepException", sidl_ex);
}

{
  tracker.writeComment("Start: Exceptions test fib-TooBigException", sidl_ex);

  var obj: exceptions.Fib = new exceptions.Fib(sidl_ex);

  init_part();
  var baseEx: BaseException = nil;
  obj.getFib(10, 1000, 1, 0, baseEx);
  writeln("baseEx = ", baseEx);
  // FIXME Need to catch the appropriate class of exception in try-catch-like block
  var isExNotNil = (baseEx != nil);
  writeln("isExNotNil = ", isExNotNil);

  run_part( "test fib-TooBigException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-TooBigException", sidl_ex);
}

tracker.close(sidl_ex);
