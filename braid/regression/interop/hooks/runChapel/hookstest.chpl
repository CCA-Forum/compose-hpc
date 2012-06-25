use hooks;
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

tracker.setExpectations(4, sidl_ex);

{ 
  var b: int(32), c: int(32), ret: int(32) = 0;
  var test: int(32) = 1;
  //FIXME! hooks.Basics_static._set_hooks_static(true);

  var obj = hooks.Basics_static.create(sidl_ex);
  //FIXME! obj._set_hooks(true);

  b = -1;
  c = -1;
  init_part(); 
  ret = hooks.Basics_static.aStaticMeth(test, b, c, sidl_ex);
  run_part( b == 1 && c == 0 );
  
  init_part(); 
  ret = hooks.Basics_static.aStaticMeth(test, b, c, sidl_ex);
  run_part( b == 2 && c == 1 );

  b = -1;
  c = -1;
  ret = obj.aNonStaticMeth(test, b, c, sidl_ex);
  run_part( b == 1 && c == 0 );
  ret = obj.aNonStaticMeth(test, b, c, sidl_ex);
  run_part( b == 2 && c == 1 );
}

tracker.close(sidl_ex);

if (failed) then
  exit(1);
