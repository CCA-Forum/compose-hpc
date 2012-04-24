
// Index is a keyword in chapel
// change sidl file entry to Employee at( in int itemIndex )

use objarg;
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

/**
 * Fill the stack with random junk.
 */
proc clearstack(magicNumber: int): int {
  return magicNumber;
}

var magicNumber = 13;
tracker.setExpectations(-1, sidl_ex);
// tracker.setExpectations(19);

proc test_Employee() { 
	
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_Employee", sidl_ex);	
  
  var numEmp = 7;
  var dataArray: [0 .. #numEmp] (string, int(32), real(32), string);
  dataArray[0] = ("John Smith", 35, 75.7e3: real(32), "c");
  dataArray[1] = ("Jane Doe", 40, 85.5e3: real(32), "m");
  dataArray[2] = ("Ella Vader", 64, 144.2e3: real(32), "r");
  dataArray[3] = ("Marge Inovera", 32, 483.2e3: real(32), "s");
  dataArray[4] = ("Hughy Louis Dewey", 45, 182.9e3: real(32), "m");
  dataArray[5] = ("Heywood Yubuzof", 12, 20.8e3: real(32), "x");
  dataArray[6] = ("Picov Andropov", 90, 120.6e3: real(32), "r");
  
  var a: objarg.EmployeeArray = objarg.EmployeeArray_static.create_EmployeeArray(sidl_ex);
  for i in [0 .. #numEmp] do {
	tracker.writeComment(" Loop.1-" + i, sidl_ex);
	
    var e: objarg.Employee = objarg.Employee_static.create_Employee(sidl_ex);
    init_part(); run_part(" init", e.init(dataArray[i][1], dataArray[i][2], dataArray[i][3], dataArray[i][4], sidl_ex));
    init_part(); run_part(" appendEmployee", a.appendEmployee(e, sidl_ex));
    init_part(); run_part(" getLength", a.getLength(sidl_ex) == (i+1));
    
    init_part(); run_part(" getAge-1.1", e.getAge(sidl_ex) == dataArray[i][2]);
    init_part(); run_part(" getSalary-1.1", e.getSalary(sidl_ex) == dataArray[i][3]);
    init_part(); run_part(" getStatus-1.1", e.getStatus(sidl_ex) == dataArray[i][4]);
    
    var e2: objarg.Employee = a.at(i+1, sidl_ex);
    init_part(); run_part(" getName-1.2", e.getName(sidl_ex) == e2.getName(sidl_ex));
    init_part(); run_part(" getAge-1.2", e.getAge(sidl_ex) == e2.getAge(sidl_ex));
    init_part(); run_part(" getSalary-1.2", e.getSalary(sidl_ex) == e2.getSalary(sidl_ex));
    init_part(); run_part(" getStatus-1.2", e.getStatus(sidl_ex) == e2.getStatus(sidl_ex));
  }
  
  for i in [0 .. #numEmp] do {
    tracker.writeComment(" Loop.2-" + i, sidl_ex);
    
    var e: objarg.Employee;
    var empInd: int(32) = a.findByName(dataArray[i][1], e, sidl_ex);
    tracker.writeComment(" emp index = " + empInd, sidl_ex);
    init_part(); run_part(" empInd", empInd == (i+1));
    if (empInd != 0) {
      var e2: objarg.Employee = a.at(empInd, sidl_ex);
      init_part(); run_part(" getName-2.1", e.getName(sidl_ex) == e2.getName(sidl_ex));
      init_part(); run_part(" getAge-2.1", e.getAge(sidl_ex) == e2.getAge(sidl_ex));
      init_part(); run_part(" getSalary-2.1", e.getSalary(sidl_ex) == e2.getSalary(sidl_ex));
      init_part(); run_part(" getStatus-2.1", e.getStatus(sidl_ex) == e2.getStatus(sidl_ex));
    }
  }
  
  /**
  var f: objarg.Employee = objarg.Employee_static.create_Employee(sidl_ex);
  f.init("Hire High", 21, 0.0: real(32), "s", sidl_ex);
  init_part(); run_part(" promoteToMaxSalary.1", a.promoteToMaxSalary(f, sidl_ex));
  init_part(); run_part(" getSalary", f.getSalary(sidl_ex) == (483.2e3: real(32)));
  init_part(); run_part(" appendEmployee", a.appendEmployee(f, sidl_ex));
  
  f = objarg.Employee_static.create_Employee(sidl_ex);
  f.init("Amadeo Avogadro, conte di Quaregna", 225, 6.022045e23: real(32), "d", sidl_ex);
  init_part(); run_part(" promoteToMaxSalary.2", !a.promoteToMaxSalary(f, sidl_ex));
  **/
  tracker.writeComment("End: test_Employee", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_Employee();

// FIXME
/**
proc test_BasicObject() { 
	
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_BasicObject", sidl_ex);	
  
  var b: objarg.Basic = objarg.Basic_static.create_Basic(sidl_ex);
  var o: sidl.BaseClass;
  var inValue: sidl.BaseClass;
  
  init_part(); run_part(" passIn", b.passIn(o, false, sidl_ex));
  
  o = sidl.BaseClass_static.create_BaseClass(sidl_ex);
  init_part(); run_part(" b.passIn-1", b.passIn(o, true, sidl_ex));
  init_part(); run_part(" b.passIn-2", b.passIn(sidl.BaseClass_static.create_BaseClass(sidl_ex), true, sidl_ex));
  
  o = sidl.BaseClass_static.create_BaseClass(sidl_ex);
  init_part(); run_part(" b.passInOut-1", b.passInOut(o, false, false, true, sidl_ex));
  // FIXME init_part(); run_part(" o._is_nil-1", o._is_nil(sidl_ex));

  o = sidl.BaseClass_static.create_BaseClass(sidl_ex);
  init_part(); run_part(" b.passInOut-2", b.passInOut(o, true, false, false, sidl_ex));
  // FIXME init_part(); run_part(" o._is_nil-2", o._is_nil(sidl_ex));
  
  o: sidl.BaseClass_static.create_BaseClass(sidl_ex);
  inValue = o;
  init_part(); run_part(" b.passInOut-3", b.passInOut(o, true, true, true, sidl_ex));
  // FIXME init_part(); run_part(" inValue.isSame", inValue.isSame(o, sidl_ex));

  o = sidl.BaseClass_static.create_BaseClass(sidl_ex);
  inValue = o;
  init_part(); run_part(" b.passInOut-4", b.passInOut(o, true, true, false, sidl_ex));
  // FIXME init_part(); run_part(" !inValue.isSame", !inValue.isSame(o, sidl_ex));
  
  o = sidl.BaseClass_static.create_BaseClass(sidl_ex);
  tracker.writeComment("b.passOut(o, false);", sidl_ex);
  b.passOut(o, false, sidl_ex);
  // FIXME init_part(); run_part(" o._not_nil", o._not_nil(sidl_ex));
  tracker.writeComment("b.passOut(o, true);", sidl_ex);
  b.passOut(o, true, sidl_ex);
  // FIXME init_part(); run_part(" o._is_nil", o._is_nil(sidl_ex));
  
  // FIXME init_part(); run_part(" b.retObject-1", b.retObject(true, sidl_ex)._is_nil(sidl_ex));
  // FIXME init_part(); run_part(" b.retObject-2", b.retObject(false, sidl_ex)._not_nil(sidl_ex));
  
  tracker.writeComment("End: test_BasicObject", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_BasicObject();
**/

tracker.close(sidl_ex);

if (failed) then
  exit(1);
