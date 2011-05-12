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
 

  /* { // char  */
  /*   char out; */
  /*   char inout = 'A'; */
  /*   Args::Basic obj = makeObject(); */
 
  /*   MYASSERT( obj.returnbackchar( ) == '3' ); */
  /*   MYASSERT( obj.passinchar( '3' ) == true ); */
  /*   MYASSERT( obj.passoutchar( out ) == true && out == '3' ); */
  /*   MYASSERT( obj.passinoutchar( inout ) == true && inout == 'a' ); */
  /*   MYASSERT( obj.passeverywherechar( '3', out, inout ) == '3' && */
  /* 	      out == '3' && inout == 'A' ); */
  /* } */


  /* { // int  */
  /*   int32_t out; */
  /*   int32_t inout = 3; */
  /*   Args::Basic obj = makeObject(); */
 
  /*   MYASSERT( obj.returnbackint( ) == 3 ); */
  /*   MYASSERT( obj.passinint( 3 ) == true ); */
  /*   MYASSERT( obj.passoutint( out ) == true && out == 3 ); */
  /*   MYASSERT( obj.passinoutint( inout ) == true && inout == -3 ); */
  /*   MYASSERT( obj.passeverywhereint( 3, out, inout ) == 3 && */
  /* 	      out == 3 && inout == 3 ); */
  /* } */


  /* { // long  */
  /*   int64_t out; */
  /*   int64_t inout = 3L; */
  /*   Args::Basic obj = makeObject(); */
 
  /*   MYASSERT( obj.returnbacklong( ) == 3L ); */
  /*   MYASSERT( obj.passinlong( 3L ) == true ); */
  /*   MYASSERT( obj.passoutlong( out ) == true && out == 3L ); */
  /*   MYASSERT( obj.passinoutlong( inout ) == true && inout == -3L ); */
  /*   MYASSERT( obj.passeverywherelong( 3L, out, inout ) == 3L && */
  /* 	      out == 3L && inout == 3L ); */
  /* } */


  /* { // float  */
  /*   ostringstream buf; */
  /*   float out; */
  /*   float inout = 3.1F; */
  /*   Args::Basic obj = makeObject(); */
  /*   buf << "obj.returnbackfloat() == " << obj.returnbackfloat(); */
  /*   tracker.writeComment(buf.str()); */
  /*   MYASSERT( obj.returnbackfloat( ) == 3.1F ); */
  /*   MYASSERT( obj.passinfloat( 3.1F ) == true ); */
  /*   MYASSERT( obj.passoutfloat( out ) == true && out == 3.1F ); */
  /*   MYASSERT( obj.passinoutfloat( inout ) == true && inout == -3.1F ); */
  /*   MYASSERT( obj.passeverywherefloat( 3.1F, out, inout ) == 3.1F && */
  /* 	      out == 3.1F && inout == 3.1F ); */
  /* } */


  /* { // double  */
  /*   double out; */
  /*   double inout = 3.14; */
  /*   Args::Basic obj = makeObject(); */
 
  /*   MYASSERT( obj.returnbackdouble( ) == 3.14 ); */
  /*   MYASSERT( obj.passindouble( 3.14 ) == true ); */
  /*   MYASSERT( obj.passoutdouble( out ) == true && out == 3.14 ); */
  /*   MYASSERT( obj.passinoutdouble( inout ) == true && inout == -3.14 ); */
  /*   MYASSERT( obj.passeverywheredouble( 3.14, out, inout ) == 3.14 && */
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
  /*   MYASSERT( retval.real() == 3.1F && retval.imag() == 3.1F); */
  /*   MYASSERT( obj.passinfcomplex( in ) == true ); */

  /*   MYASSERT( obj.passoutfcomplex( out ) == true &&  */
  /* 	      out.real() == 3.1F && out.imag() == 3.1F ); */
  /*   MYASSERT( obj.passinoutfcomplex( inout ) == true &&  */
  /* 	      inout.real() == 3.1F && inout.imag() == -3.1F ); */
  /*   tracker.writeComment("retval = obj.passeverywherefcomplex( in, out, inout );"); */
  /*   retval = obj.passeverywherefcomplex( in, out, inout ); */
  /*   MYASSERT( retval.real() == 3.1F && retval.imag() == 3.1F && */
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
  /*   MYASSERT( retval.real() == 3.14 && retval.imag() == 3.14); */
  /*   MYASSERT( obj.passindcomplex( in ) == true ); */
  /*   MYASSERT( obj.passoutdcomplex( out ) == true &&  */
  /* 	      out.real() == 3.14 && out.imag() == 3.14 ); */
  /*   MYASSERT( obj.passinoutdcomplex( inout ) == true &&  */
  /* 	      inout.real() == 3.14 && inout.imag() == -3.14 ); */
  /*   tracker.writeComment("retval = obj.passeverywheredcomplex( in, out, inout );"); */
  /*   retval = obj.passeverywheredcomplex( in, out, inout ); */
  /*   MYASSERT( retval.real() == 3.14 && retval.imag() == 3.14 && */
  /* 	      out.real() == 3.14 && out.imag() == 3.14 &&  */
  /* 	      inout.real() == 3.14 && inout.imag() == 3.14 ); */

  /* } */
  /*   } */

tracker.close();