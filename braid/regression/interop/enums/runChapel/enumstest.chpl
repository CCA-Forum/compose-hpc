use enums;
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

proc test_Enums() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_Enums", sidl_ex);
  
  init_part(); run_part(" dummy", true);
  
  tracker.writeComment("End: test_Enums", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_Enums();

proc test_colorwheel() { 
	
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_colorwheel", sidl_ex);	
  
  // undefined integer values 
  var outVar: enums.color;
  var inoutVar: enums.color = enums.color.green;
  var obj: enums.colorwheel = enums.colorwheel_static.create_colorwheel(sidl_ex);
 
  var res: enums.color = obj.returnback(sidl_ex);
  writeln(res);
  init_part(); run_part(" obj.returnback", res == enums.color.violet );
  
  init_part(); run_part(" obj.passin", obj.passin(enums.color.blue, sidl_ex));
  
  init_part(); run_part(" obj.passout-1", obj.passout(outVar, sidl_ex));
  init_part(); run_part(" obj.passout-2", outVar == enums.color.violet);
  
  init_part(); run_part(" obj.passinout-1", obj.passinout(inoutVar, sidl_ex));
  init_part(); run_part(" obj.passinout-2", inoutVar == enums.color.red);
  
  init_part(); run_part(" obj.passeverywhere-1", 
		  obj.passeverywhere(enums.color.blue, outVar, inoutVar, sidl_ex) == enums.color.violet);    
  init_part(); run_part(" obj.passeverywhere-2", outVar == enums.color.violet);
  init_part(); run_part(" obj.passeverywhere-3", inoutVar == enums.color.green);

  tracker.writeComment("End: test_colorwheel", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_colorwheel();

tracker.close(sidl_ex);
