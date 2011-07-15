use Args;
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
  tracker.writeComment("End of part "+part_no, sidl_ex);
}

// START: Assert methods
param EPSILON = 0.0001;

proc assertEquals(expected: real(32), actual: real(32)): bool 
{
  var diff = abs(expected - actual);
  var res = (diff < EPSILON);
  if (!res) {
    tracker.writeComment("Expected: " + expected + ", Found: " + actual, sidl_ex);
  }
  return res;
}

proc assertEquals(expected: real(64), actual: real(64)): bool 
{
  var diff = abs(expected - actual);
  var res = (diff < EPSILON);
  if (!res) {
    tracker.writeComment("Expected: " + expected + ", Found: " + actual, sidl_ex);
  }
  return res;
}

proc assertEquals(expected: complex(64), actual: complex(64)): bool 
{
  var res = assertEquals(expected.re, actual.re) && assertEquals(expected.im, actual.im);
  if (!res) {
    tracker.writeComment("Expected: " + expected + ", Found: " + actual, sidl_ex);
  }
  return res;
}

proc assertEquals(expected: complex(128), actual: complex(128)): bool 
{
  var res = assertEquals(expected.re, actual.re) && assertEquals(expected.im, actual.im);
  if (!res) {
    tracker.writeComment("Expected: " + expected + ", Found: " + actual, sidl_ex);
  }
  return res;
}
// END: Assert methods


{
  // bool 
  tracker.writeComment("Start: Testing bool", sidl_ex);
  var b_out: bool;
  var b_inout: bool = true;
  var obj: Args.Basic = new Args.Basic(sidl_ex);

  init_part(); run_part( "returnbackbool", obj.returnbackbool( sidl_ex) == true );
  init_part(); run_part( "passinbool", obj.passinbool( true, sidl_ex ) == true );
  init_part(); run_part( "passoutbool", obj.passoutbool( b_out, sidl_ex ) == true && b_out == true );
  init_part(); run_part( "passinoutbool", obj.passinoutbool( b_inout, sidl_ex ) == true && b_inout == false );
  init_part(); run_part( "passeverywherebool", obj.passeverywherebool( true, b_out, b_inout, sidl_ex ) == true &&
			 b_out == true && b_inout == true );    
  delete obj;
  tracker.writeComment("End: Testing bool", sidl_ex);
}

{ 
  // char
  tracker.writeComment("Start: Testing char (string in chapel)", sidl_ex);
  var c_out: string = 'DUMMY';
  var c_inout: string = 'A';
  var obj: Args.Basic = new Args.Basic(sidl_ex);
 
  local {
  init_part(); run_part( "returnbackchar", obj.returnbackchar( sidl_ex) == '3' );
  init_part(); run_part( "passinchar", obj.passinchar( '3', sidl_ex ) == true );
  init_part(); run_part( "passoutchar", obj.passoutchar( c_out, sidl_ex ) == true && c_out == '3' );
  init_part(); run_part( "passinoutchar", obj.passinoutchar( c_inout, sidl_ex ) == true && c_inout == 'a' );
  init_part(); run_part( "passeverywherechar", obj.passeverywherechar( '3', c_out, c_inout, sidl_ex ) == '3' &&
			 c_out == '3' && c_inout == 'A' );
  }
  delete obj;
  tracker.writeComment("End: Testing char (string in chapel)", sidl_ex);
}

{ 
  // int 
  tracker.writeComment("Start: Testing 32-bit int", sidl_ex);
  
  var POS_VALUE_INT32 = 3; 
  var NEG_VALUE_INT32 = -3; 

  var i32_out: int(32);
  var i32_inout: int(32) = POS_VALUE_INT32; 
  
  var obj: Args.Basic = new Args.Basic(sidl_ex);

  init_part(); run_part( obj.returnbackint(sidl_ex) == 3 );
  init_part(); run_part( obj.passinint(POS_VALUE_INT32, sidl_ex) == true );
  init_part(); run_part( obj.passoutint(i32_out, sidl_ex) == true && i32_out == POS_VALUE_INT32 );
  init_part(); run_part( obj.passinoutint(i32_inout, sidl_ex) == true && i32_inout == NEG_VALUE_INT32 );
  init_part(); run_part( obj.passeverywhereint( POS_VALUE_INT32, i32_out, i32_inout, sidl_ex) == POS_VALUE_INT32 &&
	      i32_out == POS_VALUE_INT32 && i32_inout == POS_VALUE_INT32 );
  
  tracker.writeComment("End: Testing 32-bit int", sidl_ex);
}

{ 
  // long 
  tracker.writeComment("Start: Testing 64-bit int", sidl_ex);
  
  // Large 64 bit numbers
  var THREE_POS: int(64) = 3;
  var THREE_NEG: int(64) = -3;

  var i64_in: int(64) = 3;
  var i64_out: int(64);
  var i64_inout: int(64) = 3;
  
  var obj: Args.Basic = new Args.Basic(sidl_ex);

  init_part(); run_part( obj.returnbacklong( sidl_ex) == THREE_POS );
  init_part(); run_part( obj.passinlong( i64_in, sidl_ex ) == true );
  init_part(); run_part( obj.passoutlong( i64_out, sidl_ex ) == true && i64_out == THREE_POS );
  init_part(); run_part( obj.passinoutlong( i64_inout, sidl_ex ) == true && i64_inout == THREE_NEG );
  init_part(); run_part( obj.passeverywherelong( 3, i64_out, i64_inout, sidl_ex ) == THREE_POS &&
	      i64_out == THREE_POS && i64_inout == THREE_POS );
  
  tracker.writeComment("End: Testing 64-bit int", sidl_ex);
}

//**
{ 
  // float  
  tracker.writeComment("Start: Testing 32-bit real", sidl_ex);

  var r32_out: real(32) = 0.0: real(32);  
  var r32_inout: real(32) = 3.1: real(32); 
  
  var obj: Args.Basic = new Args.Basic(sidl_ex);

  tracker.writeComment("obj.returnbackfloat() = " + obj.returnbackfloat(sidl_ex), sidl_ex);
 
  var PLUS_THREE_POINT_ONE = 3.1: real(32);
  var MINUS_THREE_POINT_ONE = -3.1: real(32);

  init_part(); run_part(assertEquals(obj.returnbackfloat(sidl_ex), PLUS_THREE_POINT_ONE));
  init_part(); run_part(obj.passinfloat( PLUS_THREE_POINT_ONE, sidl_ex ) == true );
  init_part(); run_part(obj.passoutfloat( r32_out, sidl_ex ) == true &&
          assertEquals(r32_out, PLUS_THREE_POINT_ONE)); 
  init_part(); run_part(obj.passinoutfloat( r32_inout, sidl_ex ) == true &&
          assertEquals(r32_inout, MINUS_THREE_POINT_ONE)); 
  init_part(); run_part(
          assertEquals(obj.passeverywherefloat( PLUS_THREE_POINT_ONE, r32_out, r32_inout, sidl_ex ), PLUS_THREE_POINT_ONE) &&
	      assertEquals(r32_out, PLUS_THREE_POINT_ONE) && 
          assertEquals(r32_inout, PLUS_THREE_POINT_ONE)); 

  tracker.writeComment("End: Testing 32-bit real", sidl_ex);
}
//**/

{ 
  // double  
  tracker.writeComment("Start: Testing 64-bit real", sidl_ex);

  var r64_out: real(64) = 0.0: real(64);  
  var r64_inout: real(64) = 3.14: real(64); 
  
  var obj: Args.Basic = new Args.Basic(sidl_ex);

  tracker.writeComment("obj.returnbackdouble() = " + obj.returnbackdouble(sidl_ex), sidl_ex);
 
  var PLUS_THREE_POINT_ONE_FOUR = 3.14: real(64);
  var MINUS_THREE_POINT_ONE_FOUR = -3.14: real(64);

  init_part(); run_part(assertEquals(obj.returnbackdouble(sidl_ex), PLUS_THREE_POINT_ONE_FOUR));
  init_part(); run_part( obj.passindouble( PLUS_THREE_POINT_ONE_FOUR, sidl_ex ) == true );
  init_part(); run_part( obj.passoutdouble( r64_out, sidl_ex ) == true &&
          assertEquals(r64_out, PLUS_THREE_POINT_ONE_FOUR)); 
  init_part(); run_part( obj.passinoutdouble( r64_inout, sidl_ex ) == true &&
          assertEquals(r64_inout, MINUS_THREE_POINT_ONE_FOUR)); 
  init_part(); run_part( 
          assertEquals(obj.passeverywheredouble( PLUS_THREE_POINT_ONE_FOUR, r64_out, r64_inout, sidl_ex ), PLUS_THREE_POINT_ONE_FOUR) &&
	      assertEquals(r64_out, PLUS_THREE_POINT_ONE_FOUR) && 
          assertEquals(r64_inout, PLUS_THREE_POINT_ONE_FOUR)); 

  tracker.writeComment("End: Testing 64-bit real", sidl_ex);
} 


{ 
  // complex with 32-bit floating point numbers as components
  tracker.writeComment("Start: Testing complex with 32-bit components", sidl_ex);

  var POS_VALUE_C64: complex(64); POS_VALUE_C64.re = 3.1: real(32); POS_VALUE_C64.im = 3.1: real(32); 
  var NEG_VALUE_C64: complex(64); NEG_VALUE_C64.re = 3.1: real(32); NEG_VALUE_C64.im = -3.1: real(32); 

  var retval: complex(64); 
  var c32_in: complex(64); c32_in.re = 3.1: real(32); c32_in.im = 3.1: real(32); 
  var c32_out: complex(64); 
  var c32_inout: complex(64); c32_inout.re = 3.1: real(32); c32_inout.im = 3.1: real(32); 

  var obj: Args.Basic = new Args.Basic(sidl_ex);

  //writeln("retval = " + obj.returnbackfcomplex(sidl_ex));
  
  retval = obj.returnbackfcomplex( sidl_ex);
  init_part(); run_part(assertEquals(POS_VALUE_C64, retval)); 
  init_part(); run_part( obj.passinfcomplex(c32_in, sidl_ex) == true );

  init_part(); run_part( obj.passoutfcomplex(c32_out, sidl_ex) == true &&
	      assertEquals(POS_VALUE_C64, c32_out)); 
  init_part(); run_part( obj.passinoutfcomplex(c32_inout, sidl_ex) == true &&
	      assertEquals(NEG_VALUE_C64, c32_inout)); 
  
  retval = obj.passeverywherefcomplex(c32_in, c32_out, c32_inout, sidl_ex);
  init_part(); run_part(assertEquals(POS_VALUE_C64, retval) && 
	      assertEquals(POS_VALUE_C64, c32_out) &&  
	      assertEquals(POS_VALUE_C64, c32_inout)); 
  
  tracker.writeComment("End: Testing complex with 32-bit components", sidl_ex);
}

{ 
  // complex with 64-bit floating point numbers as components
  tracker.writeComment("Start: Testing complex with 64-bit components", sidl_ex);

  var POS_VALUE_C128: complex(128); POS_VALUE_C128.re = 3.14: real(64); POS_VALUE_C128.im = 3.14: real(64); 
  var NEG_VALUE_C128: complex(128); NEG_VALUE_C128.re = 3.14: real(64); NEG_VALUE_C128.im = -3.14: real(64); 

  var retval: complex(128); 
  var c64_in: complex(128); c64_in.re = 3.14: real(64); c64_in.im = 3.14: real(64); 
  var c64_out: complex(128); 
  var c64_inout: complex(128); c64_inout.re = 3.14: real(64); c64_inout.im = 3.14: real(64); 

  var obj: Args.Basic = new Args.Basic(sidl_ex);

  //writeln("retval = " + obj.returnbackdcomplex(sidl_ex));
  
  retval = obj.returnbackdcomplex(sidl_ex);
  init_part(); run_part(assertEquals(POS_VALUE_C128, retval)); 
  init_part(); run_part( obj.passindcomplex(c64_in, sidl_ex) == true );

  init_part(); run_part( obj.passoutdcomplex(c64_out, sidl_ex) == true &&
	      assertEquals(POS_VALUE_C128, c64_out)); 
  init_part(); run_part( obj.passinoutdcomplex(c64_inout, sidl_ex) == true &&
	      assertEquals(NEG_VALUE_C128, c64_inout)); 
  
  retval = obj.passeverywheredcomplex(c64_in, c64_out, c64_inout, sidl_ex);
  init_part(); run_part(assertEquals(POS_VALUE_C128, retval) && 
	      assertEquals(POS_VALUE_C128, c64_out) &&  
	      assertEquals(POS_VALUE_C128, c64_inout)); 
  
  tracker.writeComment("End: Testing complex with 64-bit components", sidl_ex);
} 

tracker.close(sidl_ex);