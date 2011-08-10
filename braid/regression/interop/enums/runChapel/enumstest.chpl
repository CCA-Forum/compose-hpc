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

proc test_EnumsColorwheel() { 
	
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_EnumsColorwheel", sidl_ex);	
  
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

  tracker.writeComment("End: test_EnumsColorwheel", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_EnumsColorwheel();

proc test_EnumsCar() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_EnumsCar", sidl_ex);
  
  // fully defined integer values 
  var outVar: enums.car;
  var inoutVar: enums.car  = enums.car.ford;
  var obj: enums.cartest  = enums.cartest_static.create_cartest(sidl_ex);
  
  // FIXME: Need to support arrays
  // .sidl.array< .enums.car > tin, tout, tinout, tret;
   
  init_part(); run_part(" obj.returnback", obj.returnback(sidl_ex) == enums.car.porsche);
  init_part(); run_part(" obj.passin", obj.passin(enums.car.mercedes, sidl_ex));
  init_part(); run_part(" obj.passout", obj.passout(outVar, sidl_ex) && outVar == enums.car.ford);
  init_part(); run_part(" obj.passinout", obj.passinout(inoutVar, sidl_ex) && inoutVar == enums.car.porsche);
  init_part(); run_part(" obj.passeverywhere", 
      obj.passeverywhere(enums.car.mercedes, outVar, inoutVar, sidl_ex) == enums.car.porsche &&
  	  outVar == enums.car.ford && inoutVar == enums.car.mercedes);
  
  // tin = createArray();
  // tinout = createArray();
  // tracker.writeComment("Calling enums.cartest.passarray");
  // tret = obj.passarray(tin, tout, tinout);
  // init_part(); run_part(" dummy", checkArray(tin) && checkArray(tout) && checkArray(tinout) && checkArray(tret));
  
  tracker.writeComment("End: test_EnumsCar", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_EnumsCar();

proc test_EnumsNumber() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_EnumsNumber", sidl_ex);
  
  // partially defined integer values
  var outVar: enums.number;
  var inoutVar: enums.number = enums.number.zero;
  var obj: enums.numbertest = enums.numbertest_static.create_numbertest(sidl_ex);
   
  init_part(); run_part(" returnback", obj.returnback(sidl_ex) == enums.number.notOne );
  init_part(); run_part(" passin", obj.passin(enums.number.notZero, sidl_ex));
  init_part(); run_part(" passout", obj.passout(outVar, sidl_ex) && outVar == enums.number.negOne );
  init_part(); run_part(" passinout", obj.passinout(inoutVar, sidl_ex) && inoutVar == enums.number.notZero );
  init_part(); run_part(" passeverywhere", 
      obj.passeverywhere(enums.number.notZero, outVar, inoutVar, sidl_ex) == enums.number.notOne &&
  	  outVar == enums.number.negOne && inoutVar == enums.number.zero );
  
  tracker.writeComment("End: test_EnumsNumber", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_EnumsNumber();

proc test_EnumsArray() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_EnumsArray", sidl_ex);
  
  /**
   FIXME: Need to support arrays
  const int32_t numElem[] = { 2 };
  const int32_t stride[] = { 2 };
  sidl.array<enums.car> enumArray = sidl.array<enums.car>.create1d(4);
  sidl.array<enums.car> tmpArray;
  init_part(); run_part(" dummy", enumArray);
  enumArray.set(0, enums.car.porsche);
  enumArray.set(1, enums.car.ford);
  enumArray.set(2, enums.car.mercedes);
  enumArray.set(3, enums.car.porsche);
  init_part(); run_part(" dummy", enums.car.porsche == enumArray.get(0));
  init_part(); run_part(" dummy", enums.car.porsche == enumArray.get(3));
  init_part(); run_part(" dummy", enums.car.ford == enumArray.get(1));
  tmpArray = enumArray;
  init_part(); run_part(" dummy", tmpArray);
  tmpArray = tmpArray.slice(1, numElem, 0, stride);
  init_part(); run_part(" dummy", enums.car.porsche == tmpArray.get(0));
  init_part(); run_part(" dummy", enums.car.mercedes == tmpArray.get(1));
  tmpArray.smartCopy();
  init_part(); run_part(" dummy", tmpArray);
  init_part(); run_part(" dummy", enums.car.porsche == tmpArray.get(0));
  init_part(); run_part(" dummy", enums.car.mercedes == tmpArray.get(1));  
  */
  
  init_part(); run_part(" dummy", true);
  
  tracker.writeComment("End: test_EnumsArray", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_EnumsArray();

tracker.close(sidl_ex);
