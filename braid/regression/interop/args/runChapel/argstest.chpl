use Args;
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
  tracker.writeComment("End Part "+part_no);
}

{
  // bool 
  var b_out: bool;
  var b_inout: bool = true;
  var obj: Args.Basic = new Args.Basic();

  init_part(); run_part( obj.returnbackbool( ) == true );
  init_part(); run_part( obj.passinbool( true ) == true );
  init_part(); run_part( obj.passoutbool( b_out ) == true && b_out == true );
  init_part(); run_part( obj.passinoutbool( b_inout ) == true && b_inout == false );
  init_part(); run_part( obj.passeverywherebool( true, b_out, b_inout ) == true &&
			 b_out == true && b_inout == true );    
  delete obj;
}

{ 
  // char
  tracker.writeComment("Start: Testing char (string in chapel)");
  var c_out: string;
  var c_inout: string = 'A';
  var obj: Args.Basic = new Args.Basic();
 
  init_part(); run_part( obj.returnbackchar( ) == '3' );
  init_part(); run_part( obj.passinchar( '3' ) == true );
  init_part(); run_part( obj.passoutchar( c_out ) == true && c_out == '3' );
  init_part(); run_part( obj.passinoutchar( c_inout ) == true && c_inout == 'a' );
  init_part(); run_part( obj.passeverywherechar( '3', c_out, c_inout ) == '3' &&
			 c_out == '3' && c_inout == 'A' );
  delete obj;
  tracker.writeComment("End: Testing char (string in chapel)");
}

{ 
  // int 
  tracker.writeComment("Start: Testing 32-bit int");
  
  var i32_out: int(32);
  var i32_inout: int(32) = 3;
  
  var obj: Args.Basic = new Args.Basic();

  init_part(); run_part( obj.returnbackint( ) == 3 );
  init_part(); run_part( obj.passinint( 3 ) == true );
  init_part(); run_part( obj.passoutint( i32_out ) == true && i32_out == 3 );
  init_part(); run_part( obj.passinoutint( i32_inout ) == true && i32_inout == -3 );
  init_part(); run_part( obj.passeverywhereint( 3, i32_out, i32_inout ) == 3 &&
	      i32_out == 3 && i32_inout == 3 );
  
  tracker.writeComment("End: Testing 32-bit int");
}

{ 
  // long 
  tracker.writeComment("Start: Testing 64-bit int");
  
  var THREE_POS: int(64) = 3;
  var THREE_NEG: int(64) = -3;

  var i64_in: int(64) = 3;
  var i64_out: int(64);
  var i64_inout: int(64) = 3;
  
  var obj: Args.Basic = new Args.Basic();

  init_part(); run_part( obj.returnbacklong( ) == THREE_POS );
  init_part(); run_part( obj.passinlong( i64_in ) == true );
  init_part(); run_part( obj.passoutlong( i64_out ) == true && i64_out == THREE_POS );
  init_part(); run_part( obj.passinoutlong( i64_inout ) == true && i64_inout == THREE_NEG );
  init_part(); run_part( obj.passeverywherelong( 3, i64_out, i64_inout ) == THREE_POS &&
	      i64_out == THREE_POS && i64_inout == THREE_POS );
  
  tracker.writeComment("End: Testing 64-bit int");
}

//**
{ 
  // float  
  tracker.writeComment("Start: Testing 32-bit real");

  var r32_out: real(32) = 0.0: real(32);  
  var r32_inout: real(32) = 3.1: real(32); 
  
  var obj: Args.Basic = new Args.Basic();

  tracker.writeComment("obj.returnbackfloat() = " + obj.returnbackfloat());
 
  var PLUS_THREE_POINT_ONE = 3.1: real(32);
  var MINUS_THREE_POINT_ONE = -3.1: real(32);

  writeln("returnbackfloat"); init_part(); run_part( obj.returnbackfloat() == PLUS_THREE_POINT_ONE ); 
  writeln("passinfloat"); init_part(); run_part( obj.passinfloat( PLUS_THREE_POINT_ONE ) == true ); 
  writeln("passoutfloat"); init_part(); run_part( obj.passoutfloat( r32_out ) == true && r32_out == PLUS_THREE_POINT_ONE ); 
  writeln("passinoutfloat"); init_part(); run_part( obj.passinoutfloat( r32_inout ) == true && r32_inout == MINUS_THREE_POINT_ONE ); 
  writeln("passeverywherefloat"); init_part(); run_part( obj.passeverywherefloat( PLUS_THREE_POINT_ONE, r32_out, r32_inout ) == PLUS_THREE_POINT_ONE && 
	      r32_out == PLUS_THREE_POINT_ONE && r32_inout == PLUS_THREE_POINT_ONE ); 

  tracker.writeComment("End: Testing 32-bit real");
}
//**/

{ 
  // double  
  tracker.writeComment("Start: Testing 64-bit real");

  var r64_out: real(64) = 0.0: real(64);  
  var r64_inout: real(64) = 3.14: real(64); 
  
  var obj: Args.Basic = new Args.Basic();

  tracker.writeComment("obj.returnbackdouble() = " + obj.returnbackdouble());
 
  var PLUS_THREE_POINT_ONE_FOUR = 3.14: real(64);
  var MINUS_THREE_POINT_ONE_FOUR = -3.14: real(64);

  writeln("returnbackdouble"); init_part(); run_part( obj.returnbackdouble() == PLUS_THREE_POINT_ONE_FOUR ); 
  writeln("passindouble"); init_part(); run_part( obj.passindouble( PLUS_THREE_POINT_ONE_FOUR ) == true ); 
  writeln("passoutdouble"); init_part(); run_part( obj.passoutdouble( r64_out ) == true && r64_out == PLUS_THREE_POINT_ONE_FOUR ); 
  writeln("passinoutdouble"); init_part(); run_part( obj.passinoutdouble( r64_inout ) == true && r64_inout == MINUS_THREE_POINT_ONE_FOUR ); 
  writeln("passeverywheredouble"); init_part(); run_part( obj.passeverywheredouble( PLUS_THREE_POINT_ONE_FOUR, r64_out, r64_inout ) == PLUS_THREE_POINT_ONE_FOUR && 
	      r64_out == PLUS_THREE_POINT_ONE_FOUR && r64_inout == PLUS_THREE_POINT_ONE_FOUR ); 

  tracker.writeComment("End: Testing 64-bit real");
} 


  /* { // fcomplex  */
  /*   ostringstream buf; */
  /*   complex<float> retval; */
  /*   complex<float> in( 3.1F, 3.1F ); */
  /*   complex<float> out; */
  /*   complex<float> inout( 3.1F, 3.1F ); */
  /*   Args::Basic obj = makeObject(); */
 
  /*   buf << "retval = " << obj.returnbackfcomplex( ); */
  /*   tracker.writeComment(buf.str()); */
  /*   retval = obj.returnbackfcomplex( );  */
  /*   init_part(); run_part( retval.real() == 3.1F && retval.imag() == 3.1F); */
  /*   init_part(); run_part( obj.passinfcomplex( in ) == true ); */

  /*   init_part(); run_part( obj.passoutfcomplex( out ) == true &&  */
  /* 	      out.real() == 3.1F && out.imag() == 3.1F ); */
  /*   init_part(); run_part( obj.passinoutfcomplex( inout ) == true &&  */
  /* 	      inout.real() == 3.1F && inout.imag() == -3.1F ); */
  /*   tracker.writeComment("retval = obj.passeverywherefcomplex( in, out, inout );"); */
  /*   retval = obj.passeverywherefcomplex( in, out, inout ); */
  /*   init_part(); run_part( retval.real() == 3.1F && retval.imag() == 3.1F && */
  /* 	      out.real() == 3.1F && out.imag() == 3.1F &&  */
  /* 	      inout.real() == 3.1F && inout.imag() == 3.1F ); */
  /* } */


  /* { // dcomplex  */
  /*   complex<double> retval; */
  /*   complex<double> in( 3.14, 3.14 ); */
  /*   complex<double> out; */
  /*   complex<double> inout( 3.14, 3.14 ); */
  /*   Args::Basic obj = makeObject(); */
 
  /*   tracker.writeComment("retval = obj.returnback( );"); */
  /*   retval = obj.returnbackdcomplex( ); */
  /*   init_part(); run_part( retval.real() == 3.14 && retval.imag() == 3.14); */
  /*   init_part(); run_part( obj.passindcomplex( in ) == true ); */
  /*   init_part(); run_part( obj.passoutdcomplex( out ) == true &&  */
  /* 	      out.real() == 3.14 && out.imag() == 3.14 ); */
  /*   init_part(); run_part( obj.passinoutdcomplex( inout ) == true &&  */
  /* 	      inout.real() == 3.14 && inout.imag() == -3.14 ); */
  /*   tracker.writeComment("retval = obj.passeverywheredcomplex( in, out, inout );"); */
  /*   retval = obj.passeverywheredcomplex( in, out, inout ); */
  /*   init_part(); run_part( retval.real() == 3.14 && retval.imag() == 3.14 && */
  /* 	      out.real() == 3.14 && out.imag() == 3.14 &&  */
  /* 	      inout.real() == 3.14 && inout.imag() == 3.14 ); */

  /* } */
  /*   } */

tracker.close();