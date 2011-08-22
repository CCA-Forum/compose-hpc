use Overload;
use synch;
use sidl;

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

proc test_Overload() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_Overload", sidl_ex);
  
  var b1: bool = true;
  var d1: real(64) = 1.0;
  var f1: real(32) = 1.0: real(32);
  var i1: int(32) = 1;
  var did: real(64) = 2.0;
  var difd: real(64) = 3.0;
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
  
  cdret = t.getValueDcomplex(cd1, sidl_ex);
  init_part(); run_part(" t.getValueDcomplex(cd1).re", cdret.re == cd1.re);
  init_part(); run_part(" t.getValueDcomplex(cd1).im", cdret.im == cd1.im);
  
  init_part(); run_part(" t.getValue(f1)", t.getValueFloat(f1, sidl_ex) == f1);
  
  cfret = t.getValueFcomplex(cf1, sidl_ex);
  init_part(); run_part(" t.getValueFcomplex(cf1).re", cfret.re == cf1.re);
  init_part(); run_part(" t.getValueFcomplex(cf1).im", cfret.im == cf1.im);
  
  init_part(); run_part(" t.getValueInt(i1)", t.getValueInt(i1, sidl_ex) == i1);
  init_part(); run_part(" t.getValueString(s1)", t.getValueString(s1, sidl_ex) == s1);

  init_part(); run_part(" t.getValueDoubleInt(d1, i1)", t.getValueDoubleInt(d1, i1, sidl_ex) == did);
  init_part(); run_part(" t.getValueIntDouble(i1, d1)", t.getValueIntDouble(i1, d1, sidl_ex) == did);

  init_part(); run_part(" t.getValueDoubleIntFloat(d1, i1, f1)", t.getValueDoubleIntFloat(d1, i1, f1, sidl_ex) == difd);
  init_part(); run_part(" t.getValueIntDoubleFloat(i1, d1, f1)", t.getValueIntDoubleFloat(i1, d1, f1, sidl_ex) == difd);

  init_part(); run_part(" t.getValueDoubleFloatInt(d1, f1, i1)", t.getValueDoubleFloatInt(d1, f1, i1, sidl_ex) == difd);
  init_part(); run_part(" t.getValueIntFloatDouble(i1, f1, d1)", t.getValueIntFloatDouble(i1, f1, d1, sidl_ex) == difd);

  init_part(); run_part(" t.getValueFloatDoubleInt(f1, d1, i1)", t.getValueFloatDoubleInt(f1, d1, i1, sidl_ex) == difd);
  init_part(); run_part(" t.getValueFloatIntDouble(f1, i1, d1)", t.getValueFloatIntDouble(f1, i1, d1, sidl_ex) == difd);

  init_part(); run_part(" t.getValueException(ae)", t.getValueException(ae, sidl_ex) == "AnException");
  init_part(); run_part(" t.getValueAClass(ac)", t.getValueAClass(ac, sidl_ex) == 2);
  init_part(); run_part(" t.getValueBClass(bc)", t.getValueBClass(bc, sidl_ex) == 2);
  
  tracker.writeComment("End: test_Overload", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_Overload();

tracker.close(sidl_ex);

if (failed) then
  exit(1);
