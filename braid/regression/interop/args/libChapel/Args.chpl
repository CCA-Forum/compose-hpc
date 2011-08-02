// 
// File:          Args_Basic_Impl.cxx
// Symbol:        Args.Basic-v1.0
// Symbol Type:   class
// Babel Version: 1.5.0 (Revision: 6763M trunk)
// Description:   Server-side implementation for Args.Basic
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

// DO-NOT-DELETE splicer.begin(Args.Basic._includes)
// Insert-Code-Here {Args.Basic._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(Args.Basic._includes)

class Basic_Impl {

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
//Basic_impl() : StubBase(reinterpret_cast< void*>(
//  ::Args::Basic::_wrapObj(reinterpret_cast< void*>(this))),false) , _wrapped(
//  true){ 
  // DO-NOT-DELETE splicer.begin(Args.Basic._ctor2)
  // Insert-Code-Here {Args.Basic._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(Args.Basic._ctor2)
  //}

  // user defined constructor
  proc _ctor() {
    // DO-NOT-DELETE splicer.begin(Args.Basic._ctor)
    // Insert-Code-Here {Args.Basic._ctor} (constructor)
    // DO-NOT-DELETE splicer.end(Args.Basic._ctor)
  }
   
  // user defined destructor
  proc _dtor() {
    // DO-NOT-DELETE splicer.begin(Args.Basic._dtor)
    // Insert-Code-Here {Args.Basic._dtor} (destructor)
    // DO-NOT-DELETE splicer.end(Args.Basic._dtor)
  }
   
  // static class initializer
  proc _load() {
    // DO-NOT-DELETE splicer.begin(Args.Basic._load)
    // Insert-Code-Here {Args.Basic._load} (class initialization)
    // DO-NOT-DELETE splicer.end(Args.Basic._load)
  }

  // user defined static methods: (none)
   
  // user defined non-static methods:
  /**
   * Method:  returnbackbool[]
   */
  proc
  returnbackbool_impl() : bool
   
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackbool)
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackbool)
  }
   
  /**
   * Method:  passinbool[]
   */
  proc
  passinbool_impl( in b: bool ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinbool)
    return b;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinbool)
  }
   
  /**
   * Method:  passoutbool[]
   */
  proc
  passoutbool_impl( out b: bool ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutbool)
    b = true;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutbool)
  }
   
  /**
   * Method:  passinoutbool[]
   */
  proc
  passinoutbool_impl( inout b: bool ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutbool)
    b = !b;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutbool)
  }
   
  /**
   * Method:  passeverywherebool[]
   */
  proc
  passeverywherebool_impl( in b1: bool,
    out b2: bool,
    inout b3: bool ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherebool)
    b2 = true;
    b3 = !b3;
    return b1;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywherebool)
  }
   
  /**
   * Method:  returnbackchar[]
   */
  proc
  returnbackchar_impl() : string
   
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackchar)
    return '3';
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackchar)
  }
   
  /**
   * Method:  passinchar[]
   */
  proc
  passinchar_impl( in c: string ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinchar)
      return( c == '3');
    // DO-NOT-DELETE splicer.end(Args.Basic.passinchar)
  }
   
  /**
   * Method:  passoutchar[]
   */
  proc
  passoutchar_impl( out c: string ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutchar)
    c = '3';
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutchar)
  }
   
  /**
   * Method:  passinoutchar[]
   */
  proc
  passinoutchar_impl( inout c: string ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutchar)
    if ( c >= 'a' && c <= 'z' ) { 
      //c += 'A' - 'a';
    } else if ( c >= 'A' && c <= 'Z' ) { 
      //c += 'a' - 'A';
    }
    return true;  
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutchar)
  }
   
  /**
   * Method:  passeverywherechar[]
   */
  proc
  passeverywherechar_impl( in c1: string,
    out c2: string,
    inout c3: string ) : string 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherechar)
    c2 = '3';
    if ( c3 >= 'a' && c3 <= 'z' ) { 
      //c3 += 'A' - 'a';
    } else if ( c3 >= 'A' && c3 <= 'Z' ) { 
      //c3 += 'a' - 'A';
    }
    if ( c1 == '3') then
      return '3';
    else return  '\0';
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywherechar)
  }
   
  /**
   * Method:  returnbackint[]
   */
  proc
  returnbackint_impl() : int(32)
   
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackint)
    return 3;
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackint)
  }
   
  /**
   * Method:  passinint[]
   */
  proc
  passinint_impl( in i: int(32) ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinint)
    return ( i == 3 );
    // DO-NOT-DELETE splicer.end(Args.Basic.passinint)
  }
   
  /**
   * Method:  passoutint[]
   */
  proc
  passoutint_impl( out i: int(32) ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutint)
    i = 3;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutint)
  }
   
  /**
   * Method:  passinoutint[]
   */
  proc
  passinoutint_impl( inout i: int(32) ) : bool 
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutint)
    i = -i;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutint)
  }
   
  /**
   * Method:  passeverywhereint[]
   */
  proc
  passeverywhereint_impl( in i1: int(32),
			  out i2: int(32),
			  inout i3: int(32)) : int(32)
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywhereint)
    i2 = 3;
    i3 = -i3;
    if ( i1 == 3 ) then 
      return 3 ;
    else return 0 ;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywhereint)
  }
   
  /**
   * Method:  returnbacklong[]
   */
  proc
  returnbacklong_impl() : int(64)
   
  {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbacklong)
    return 3;
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbacklong)
  }
   
  /**
   * Method:  passinlong[]
   */
  proc passinlong_impl( in l: int(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinlong)
    return ( l == 3 );
    // DO-NOT-DELETE splicer.end(Args.Basic.passinlong)
  }
   
  /**
   * Method:  passoutlong[]
   */
    proc passoutlong_impl( out l: int(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutlong)
    l = 3;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutlong)
  }
   
  /**
   * Method:  passinoutlong[]
   */
  proc passinoutlong_impl( inout l: int(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutlong)
    l = -l;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutlong)
  }
   
  /**
   * Method:  passeverywherelong[]
   */
  proc passeverywherelong_impl( in l1: int(64), out l2: int(64), inout l3: int(64)): int(64) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherelong)
    l2 = 3;
    l3 = -l3;
    if ( l1 == 3 ) then return 3; else return 0;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywherelong)
  }
   
  /**
   * Method:  returnbackfloat[]
   */
  proc returnbackfloat_impl(): real(32) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfloat)
    return 3.1:real(32);
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackfloat)
  }
   
  /**
   * Method:  passinfloat[]
   */
  proc passinfloat_impl( in f: real(32)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinfloat)
     return ( f == 3.1:real(32) );
    // DO-NOT-DELETE splicer.end(Args.Basic.passinfloat)
  }
   
  /**
   * Method:  passoutfloat[]
   */
  proc passoutfloat_impl( out f: real(32)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutfloat)
    f = 3.1:real(32);
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutfloat)
  }
   
  /**
   * Method:  passinoutfloat[]
   */
  proc passinoutfloat_impl( inout f: real(32)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfloat)
    f = -f;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutfloat)
  }
   
  /**
   * Method:  passeverywherefloat[]
   */
  proc passeverywherefloat_impl( in f1: real(32), out f2: real(32), inout f3: real(32)): real(32) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefloat)
    f2 = 3.1:real(32);
    f3 = -f3;
    if ( f1 == 3.1:real(32) ) then return 3.1:real(32); else return 0.0:real(32);
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefloat)
  }
   
  /**
   * Method:  returnbackdouble[]
   */
  proc returnbackdouble_impl(): real(64) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdouble)
    return 3.14;
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackdouble)
  }
   
  /**
   * Method:  passindouble[]
   */
  proc passindouble_impl( in d: real(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passindouble)
    return ( d == 3.14 );
    // DO-NOT-DELETE splicer.end(Args.Basic.passindouble)
  }
   
  /**
   * Method:  passoutdouble[]
   */
  proc passoutdouble_impl( out d: real(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutdouble)
    d = 3.14;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutdouble)
  }
   
  /**
   * Method:  passinoutdouble[]
   */
  proc passinoutdouble_impl( inout d: real(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdouble)
    d = -d;
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutdouble)
  }
   
  /**
   * Method:  passeverywheredouble[]
   */
  proc passeverywheredouble_impl( in d1: real(64), out d2: real(64), inout d3: real(64)): real(64) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredouble)
    d2 = 3.14;
    d3 = -d3;
    if ( d1 == 3.14 ) then return 3.14; else return 0.0 ;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredouble)
  }
   
  /**
   * Method:  returnbackfcomplex[]
   */
  proc returnbackfcomplex_impl(): complex(64) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfcomplex)
    return (3.1 + 3.1i):complex(64);
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackfcomplex)
  }
   
  /**
   * Method:  passinfcomplex[]
   */
  proc passinfcomplex_impl( in c: complex(64)): bool {

    // DO-NOT-DELETE splicer.begin(Args.Basic.passinfcomplex)
    return ( c.re == 3.1 && c.im == 3.1 );
    // DO-NOT-DELETE splicer.end(Args.Basic.passinfcomplex)
  }
   
  /**
   * Method:  passoutfcomplex[]
   */
  proc passoutfcomplex_impl( out c: complex(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutfcomplex)
    c = (3.1 + 3.1i):complex(64);
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutfcomplex)
  }
   
  /**
   * Method:  passinoutfcomplex[]
   */
  proc passinoutfcomplex_impl( inout c: complex(64)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfcomplex)
    c = (c.re,-c.im):complex(64);
    // was 'c=conj(c)', but I've had too many compilers complain... :[
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutfcomplex)
  }
   
  /**
   * Method:  passeverywherefcomplex[]
   */
  proc passeverywherefcomplex_impl( in c1: complex(64), out c2: complex(64), inout c3: complex(64)): complex(64) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefcomplex)
    var temp1: complex(64) = (3.1,3.1):complex(64);
    var error: complex(64) = (0.0, 0.0):complex(64);
    c2 = temp1;
    c3 = (c3.re,-c3.im):complex(64);
    // was 'c3=conj(c3)', but I've had too many compilers complain... :[
    if ( c1 == temp1 ) then return temp1; else return error ;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefcomplex)
  }
   
  /**
   * Method:  returnbackdcomplex[]
   */
  proc returnbackdcomplex_impl(): complex(128) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdcomplex)
    return (3.14, 3.14):complex(128);
    // DO-NOT-DELETE splicer.end(Args.Basic.returnbackdcomplex)
  }
   
  /**
   * Method:  passindcomplex[]
   */
  proc passindcomplex_impl( in c: complex(128)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passindcomplex)
    return ( c.re == 3.14 && c.im == 3.14 );
    // DO-NOT-DELETE splicer.end(Args.Basic.passindcomplex)
  }
   
  /**
   * Method:  passoutdcomplex[]
   */
  proc passoutdcomplex_impl( out c: complex(128)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passoutdcomplex)
    c = (3.14,3.14):complex(128);
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passoutdcomplex)
  }
   
  /**
   * Method:  passinoutdcomplex[]
   */
  proc passinoutdcomplex_impl( inout c: complex(128)): bool {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdcomplex)
    c = (c.re, -c.im):complex(128);
    // was 'c=conj(c);' but I've had too many compilers complain for no reason.  :[
    return true;
    // DO-NOT-DELETE splicer.end(Args.Basic.passinoutdcomplex)
  }
   
  /**
   * Method:  passeverywheredcomplex[]
   */
  proc passeverywheredcomplex_impl( in c1: complex(128), out c2: complex(128), inout c3: complex(128)): complex(128) {
    // DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredcomplex)
    var temp : complex(128) = ( 3.14, 3.14 ):complex(128);
    var error: complex(128) = (0.0,0.0):complex(128);
    c2 = temp;
    c3 = (c3.re,-c3.im):complex(128); 
    // was 'c3=conj(c3);' but I've had too many compilers complain for no reason.  :[
    if ( c1 == temp ) then return temp; else return error;
    // DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredcomplex)
  }


// DO-NOT-DELETE splicer.begin(Args.Basic._misc)
// Insert-Code-Here {Args.Basic._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Args.Basic._misc)

}

//use Args_Skel;
//Args_Skel.__defeat_dce();
var obj = new Basic_Impl();
obj._ctor();
obj._dtor();
obj._load();
{
obj.returnbackbool_impl( );
}
;
{
var b:bool;
obj.passinbool_impl( b);
}
;
{
var b:bool;
obj.passoutbool_impl( b);
}
;
{
var b:bool;
obj.passinoutbool_impl( b);
}
;
{
var b1:bool;
var b2:bool;
var b3:bool;
obj.passeverywherebool_impl( b1, b2, b3);
}
;
{
obj.returnbackchar_impl( );
}
;
{
var c:string;
obj.passinchar_impl( c);
}
;
{
var c:string;
obj.passoutchar_impl( c);
}
;
{
var c:string;
obj.passinoutchar_impl( c);
}
;
{
var c1:string;
var c2:string;
var c3:string;
obj.passeverywherechar_impl( c1, c2, c3);
}
;
{
obj.returnbackint_impl( );
}
;
{
var i:int(32);
obj.passinint_impl( i);
}
;
{
var i:int(32);
obj.passoutint_impl( i);
}
;
{
var i:int(32);
obj.passinoutint_impl( i);
}
;
{
var i1:int(32);
var i2:int(32);
var i3:int(32);
obj.passeverywhereint_impl( i1, i2, i3);
}
;
{
obj.returnbacklong_impl( );
}
;
{
var l:int(64);
obj.passinlong_impl( l);
}
;
{
var l:int(64);
obj.passoutlong_impl( l);
}
;
{
var l:int(64);
obj.passinoutlong_impl( l);
}
;
{
var l1:int(64);
var l2:int(64);
var l3:int(64);
obj.passeverywherelong_impl( l1, l2, l3);
}
;
{
obj.returnbackfloat_impl( );
}
;
{
var f:real(32);
obj.passinfloat_impl( f);
}
;
{
var f:real(32);
obj.passoutfloat_impl( f);
}
;
{
var f:real(32);
obj.passinoutfloat_impl( f);
}
;
{
var f1:real(32);
var f2:real(32);
var f3:real(32);
obj.passeverywherefloat_impl( f1, f2, f3);
}
;
{
obj.returnbackdouble_impl( );
}
;
{
var d:real(64);
obj.passindouble_impl( d);
}
;
{
var d:real(64);
obj.passoutdouble_impl( d);
}
;
{
var d:real(64);
obj.passinoutdouble_impl( d);
}
;
{
var d1:real(64);
var d2:real(64);

var d3:real(64);
obj.passeverywheredouble_impl( d1, d2, d3);
}
;
{
obj.returnbackfcomplex_impl( );
}
;
{
var c:complex(64);
obj.passinfcomplex_impl( c);
}
;
{
var c:complex(64);
obj.passoutfcomplex_impl( c);
}
;
{
var c:complex(64);
obj.passinoutfcomplex_impl( c);
}
;
{
var c1:complex(64);
var c2:complex(64);
var c3:complex(64);
obj.passeverywherefcomplex_impl( c1, c2, c3);
}
;
{
obj.returnbackdcomplex_impl( );
}
;
{
var c:complex(128);
obj.passindcomplex_impl( c);
}
;
{
var c:complex(128);
obj.passoutdcomplex_impl( c);
}
;
{
var c:complex(128);
obj.passinoutdcomplex_impl( c);
}
;
{
var c1:complex(128);
var c2:complex(128);
var c3:complex(128);
obj.passeverywheredcomplex_impl( c1, c2, c3);
}
;
