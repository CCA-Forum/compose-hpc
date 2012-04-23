use Exceptions;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

var failed: bool = false;
var part_no: int(32) = 0;
var sidl_ex: BaseInterface = nil;
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

  var obj: Exceptions.Fib = Exceptions.Fib_static.create_Fib(sidl_ex);

  init_part();
  var baseEx: BaseInterface = nil;
  var isExNotNil = (baseEx != nil);
  obj.getFib(10, 25, 200, 0, baseEx);
  isExNotNil = (baseEx != nil);
  run_part( "test fib-good", baseEx == nil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-good", sidl_ex);
}
writeln();
{
  tracker.writeComment("Start: Exceptions test fib-NegativeValueException", sidl_ex);

  var obj: Exceptions.Fib = Exceptions.Fib_static.create_Fib(sidl_ex);

  init_part();
  var baseEx: BaseInterface = nil;
  obj.getFib(-1, 10, 10, 0, baseEx);
  writeln("Expecting NegativeValueException, found baseEx = ", baseEx);
  var isExNotNil = (baseEx != nil);
  run_part( "test fib-NegativeValueException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-NegativeValueException", sidl_ex);
}
writeln();
{
  tracker.writeComment("Start: Exceptions test fib-TooDeepException", sidl_ex);

  var obj: Exceptions.Fib = Exceptions.Fib_static.create_Fib(sidl_ex);

  init_part();
  var baseEx: BaseInterface = nil;
  obj.getFib(10, 1, 1000, 0, baseEx);
  writeln("Expecting TooDeepException, found baseEx = ", baseEx);
  // FIXME Need to catch the appropriate class of exception in try-catch-like block
  var isExNotNil = (baseEx != nil);  
  run_part( "test fib-TooDeepException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-TooDeepException", sidl_ex);
}
writeln();
{
  tracker.writeComment("Start: Exceptions test fib-TooBigException", sidl_ex);

  var obj: Exceptions.Fib = Exceptions.Fib_static.create_Fib(sidl_ex);

  init_part();
  var baseEx: BaseInterface = nil;
  obj.getFib(10, 1000, 1, 0, baseEx);
  writeln("Expecting TooBigException, found baseEx = ", baseEx);
  // FIXME Need to catch the appropriate class of exception in try-catch-like block
  var isExNotNil = (baseEx != nil);
  run_part( "test fib-TooBigException", isExNotNil );

  delete obj;
  tracker.writeComment("End: Exceptions test fib-TooBigException", sidl_ex);
}
writeln();

tracker.close(sidl_ex);

if (failed) then
  exit(1);
