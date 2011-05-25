// 
// File:        stringstests.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Test string interoperability
// 
use Strings;
use synch;

var part_no: int = 0;
var tracker: synch.RegOut = synch.getInstance();

proc init_part()
{
  part_no += 1;
  tracker.startPart(part_no);
  tracker.writeComment("Part "+part_no);
}

proc run_part(result: bool)
{
  var r: ResultType;
  if (result) then
    r = ResultType.PASS;
  else 
    r = ResultType.FAIL;
  tracker.endPart(part_no, r);
  tracker.writeComment("End of part "+part_no);
}

var obj = new Strings.Cstring();
tracker.setExpectations(-1);
  
{ 
  var s_in: string= "Three";
  var s_out: string;
  var s_inout: string = "Three";

  init_part(); run_part( obj.returnback(true) == "Three" );
  init_part(); run_part( obj.returnback(false) == "" );
  init_part(); run_part( obj.passin( s_in ) == true );
  init_part(); run_part( obj.passout(true, s_out ) == true && s_out == "Three" );
  init_part(); run_part( obj.passout(false, s_out ) == false && s_out == "" );
  init_part(); run_part( obj.passinout( s_inout ) == true && s_inout == "threes" );
  init_part(); run_part( obj.passeverywhere( s_in, s_out, s_inout ) == "Three" &&
      s_out == "Three" && s_inout == "Three" );
  init_part(); run_part( obj.mixedarguments( "Test", 'z', "Test", 'z') );
  init_part(); run_part( !obj.mixedarguments( "Not", 'A', "Equal", 'a') );
  
}

tracker.close();
