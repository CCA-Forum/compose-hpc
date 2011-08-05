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

proc test_Overload() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_Overload", sidl_ex);
  
  var b1: bool = true;
  var d1: real(64) = 1.0;
  var f1: real(32) = 1.0: real(32);
  var i1: int(32) = 1;
  var did: real(64) = 2.0;
  var dfid: real(64) = 3.0;
  var s1: string = "aString";
  
  var cd1: complex(128); cd1.re =1.1; cd1.im = 1.1;
  var cdret: complex(128);
  var cf1: complex(64); cf1.re =1.1: real(32); cf1.im = 1.1: real(32);
  var cfret: complex(64);
    
  var t:  Overload.Test         = Overload.Test_static.create_Test(sidl_ex);
  var ae: Overload.AnException  = Overload.AnException_static.create_AnException(sidl_ex);
  var ac: Overload.AClass       = Overload.AClass_static.create_AClass(sidl_ex);
  var bc: Overload.BClass       = Overload.BClass_static.create_BClass(sidl_ex);
  
  init_part(); run_part(" t.getValue", t.getValue(sidl_ex) == 1);
  
  init_part(); run_part(" t.getValue(b1)", t.getValueBool(b1, sidl_ex) == b1);
  init_part(); run_part(" t.getValue(d1)", t.getValueDouble(d1, sidl_ex) == d1);
  
  tracker.writeComment("End: test_Overload", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_Overload();

tracker.close(sidl_ex);
