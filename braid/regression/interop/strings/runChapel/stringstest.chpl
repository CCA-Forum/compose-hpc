// 
// File:        stringstests.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Test string interoperability
// 
use Strings;
use synch;

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

var obj = Strings.Cstring_static.create_Cstring(sidl_ex);
tracker.setExpectations(-1, sidl_ex);
  
{ 
  var s_in: string= "Three";
  var s_out: string;
  var s_inout: string = "Three";

  init_part(); run_part( obj.returnback(true, sidl_ex) == "Three" );
  init_part(); run_part( obj.returnback(false, sidl_ex) == "" );
  init_part(); run_part( obj.passin( s_in, sidl_ex ) == true );
  init_part(); run_part( obj.passout(true, s_out, sidl_ex ) == true && s_out == "Three" );
  init_part(); run_part( obj.passout(false, s_out, sidl_ex ) == false && s_out == "" );
  init_part(); run_part( obj.passinout( s_inout, sidl_ex ) == true && s_inout == "threes" );
  init_part(); run_part( obj.passeverywhere( s_in, s_out, s_inout, sidl_ex ) == "Three" &&
      s_out == "Three" && s_inout == "Three" );
  init_part(); run_part( obj.mixedarguments( "Test", 'z', "Test", 'z', sidl_ex) );
  init_part(); run_part( !obj.mixedarguments( "Not", 'A', "Equal", 'a', sidl_ex) );
  
}

tracker.close(sidl_ex);

if (failed) then
  exit(1);
