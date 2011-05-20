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
  tracker.writeComment("Part "+part_no);
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
  writeln("Start: Testing char (string in chapel)");
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
  writeln("End: Testing char (string in chapel)");
}

{ 
  // int 
  writeln("Start: Testing 32-bit int");
  
  var i32_out: int(32);
  var i32_inout: int(32) = 3;
  
  var obj: Args.Basic = new Args.Basic();

  init_part(); run_part( obj.returnbackint( ) == 3 );
  init_part(); run_part( obj.passinint( 3 ) == true );
  init_part(); run_part( obj.passoutint( i32_out ) == true && i32_out == 3 );
  init_part(); run_part( obj.passinoutint( i32_inout ) == true && i32_inout == -3 );
  init_part(); run_part( obj.passeverywhereint( 3, i32_out, i32_inout ) == 3 &&
	      i32_out == 3 && i32_inout == 3 );
  
  writeln("End: Testing 32-bit int");
}

{ 
  // long 
  writeln("Start: Testing 64-bit int");
  
  var THREE_POS: int(64) = 3;
  var THREE_NEG: int(64) = -3;

  var i64_in: int(64) = 3;
  var i64_out: int(64);
  var i64_inout: int(64) = 3;
  
  var obj: Args.Basic = new Args.Basic();

  writeln('returnbacklong'); init_part(); run_part( obj.returnbacklong( ) == THREE_POS );
  writeln('passinlong'); init_part(); run_part( obj.passinlong( i64_in ) == true );
  writeln('passoutlong'); init_part(); run_part( obj.passoutlong( i64_out ) == true && i64_out == THREE_POS );
  writeln('passinoutlong'); init_part(); run_part( obj.passinoutlong( i64_inout ) == true && i64_inout == THREE_NEG );
  writeln('passeverywherelong'); init_part(); run_part( obj.passeverywherelong( 3, i64_out, i64_inout ) == THREE_POS &&
	      i64_out == THREE_POS && i64_inout == THREE_POS );
  
  writeln("End: Testing 64-bit int");
}

  /* { // float  */
  /*   ostringstream buf; */
  /*   float out; */
  /*   float inout = 3.1F; */
  /*   Args::Basic obj = makeObject(); */
  /*   buf << "obj.returnbackfloat() == " << obj.returnbackfloat(); */
  /*   tracker.writeComment(buf.str()); */
  /*   init_part(); run_part( obj.returnbackfloat( ) == 3.1F ); */
  /*   init_part(); run_part( obj.passinfloat( 3.1F ) == true ); */
  /*   init_part(); run_part( obj.passoutfloat( out ) == true && out == 3.1F ); */
  /*   init_part(); run_part( obj.passinoutfloat( inout ) == true && inout == -3.1F ); */
  /*   init_part(); run_part( obj.passeverywherefloat( 3.1F, out, inout ) == 3.1F && */
  /* 	      out == 3.1F && inout == 3.1F ); */
  /* } */


  /* { // double  */
  /*   double out; */
  /*   double inout = 3.14; */
  /*   Args::Basic obj = makeObject(); */
 
  /*   init_part(); run_part( obj.returnbackdouble( ) == 3.14 ); */
  /*   init_part(); run_part( obj.passindouble( 3.14 ) == true ); */
  /*   init_part(); run_part( obj.passoutdouble( out ) == true && out == 3.14 ); */
  /*   init_part(); run_part( obj.passinoutdouble( inout ) == true && inout == -3.14 ); */
  /*   init_part(); run_part( obj.passeverywheredouble( 3.14, out, inout ) == 3.14 && */
  /* 	      out == 3.14 && inout == 3.14 ); */
  /* } */


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