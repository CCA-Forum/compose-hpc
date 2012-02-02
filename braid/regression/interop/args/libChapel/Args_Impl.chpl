use sidl;
use Args;

export returnbackbool_impl proc returnbackbool_impl( in _this: opaque): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackbool) */
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackbool) */
}


export passinbool_impl proc passinbool_impl( in _this: opaque, in b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinbool) */
    return b;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinbool) */
}


export passoutbool_impl proc passoutbool_impl( in _this: opaque, out b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutbool) */
    b = true;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutbool) */
}


export passinoutbool_impl proc passinoutbool_impl( in _this: opaque, inout b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutbool) */
    b = !b;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutbool) */
}


export passeverywherebool_impl proc passeverywherebool_impl( in _this: opaque, in b1: bool, out b2: bool, inout b3: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherebool) */
    b2 = true;
    b3 = !b3;
    return b1;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherebool) */
}


export returnbackchar_impl proc returnbackchar_impl( in _this: opaque): string {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackchar) */
    return '3';
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackchar) */
}


export passinchar_impl proc passinchar_impl( in _this: opaque, in c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinchar) */
      return( c == '3');
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinchar) */
}


export passoutchar_impl proc passoutchar_impl( in _this: opaque, out c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutchar) */
    c = '3';
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutchar) */
}


export passinoutchar_impl proc passinoutchar_impl( in _this: opaque, inout c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutchar) */
    if ( c >= 'a' && c <= 'z' ) { 
      //c += 'A' - 'a';
    } else if ( c >= 'A' && c <= 'Z' ) { 
      //c += 'a' - 'A';
    }
    return true;  
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutchar) */
}


export passeverywherechar_impl proc passeverywherechar_impl( in _this: opaque, in c1: string, out c2: string, inout c3: string): string {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherechar) */
    c2 = '3';
    if ( c3 >= 'a' && c3 <= 'z' ) { 
      //c3 += 'A' - 'a';
    } else if ( c3 >= 'A' && c3 <= 'Z' ) { 
      //c3 += 'a' - 'A';
    }
    if ( c1 == '3') then
      return '3';
    else return  '\0';
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherechar) */
}


export returnbackint_impl proc returnbackint_impl( in _this: opaque): int(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackint) */
    return 3;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackint) */
}


export passinint_impl proc passinint_impl( in _this: opaque, in i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinint) */
    return ( i == 3 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinint) */
}


export passoutint_impl proc passoutint_impl( in _this: opaque, out i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutint) */
    i = 3;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutint) */
}


export passinoutint_impl proc passinoutint_impl( in _this: opaque, inout i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutint) */
    i = -i;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutint) */
}


export passeverywhereint_impl proc passeverywhereint_impl( in _this: opaque, in i1: int(32), out i2: int(32), inout i3: int(32)): int(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywhereint) */
    i2 = 3;
    i3 = -i3;
    if ( i1 == 3 ) then 
      return 3 ;
    else return 0 ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywhereint) */
}


export returnbacklong_impl proc returnbacklong_impl( in _this: opaque): int(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbacklong) */
    return 3;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbacklong) */
}


export passinlong_impl proc passinlong_impl( in _this: opaque, in l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinlong) */
    return ( l == 3 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinlong) */
}


export passoutlong_impl proc passoutlong_impl( in _this: opaque, out l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutlong) */
    l = 3;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutlong) */
}


export passinoutlong_impl proc passinoutlong_impl( in _this: opaque, inout l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutlong) */
    l = -l;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutlong) */
}


export passeverywherelong_impl proc passeverywherelong_impl( in _this: opaque, in l1: int(64), out l2: int(64), inout l3: int(64)): int(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherelong) */
    l2 = 3;
    l3 = -l3;
    if ( l1 == 3 ) then return 3; else return 0;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherelong) */
}


export returnbackfloat_impl proc returnbackfloat_impl( in _this: opaque): real(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfloat) */
    return 3.1:real(32);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackfloat) */
}


export passinfloat_impl proc passinfloat_impl( in _this: opaque, in f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinfloat) */
     return ( f == 3.1:real(32) );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinfloat) */
}


export passoutfloat_impl proc passoutfloat_impl( in _this: opaque, out f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutfloat) */
    f = 3.1:real(32);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutfloat) */
}


export passinoutfloat_impl proc passinoutfloat_impl( in _this: opaque, inout f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfloat) */
    f = -f;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutfloat) */
}


export passeverywherefloat_impl proc passeverywherefloat_impl( in _this: opaque, in f1: real(32), out f2: real(32), inout f3: real(32)): real(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefloat) */
    f2 = 3.1:real(32);
    f3 = -f3;
    if ( f1 == 3.1:real(32) ) then return 3.1:real(32); else return 0.0:real(32);
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefloat) */
}


export returnbackdouble_impl proc returnbackdouble_impl( in _this: opaque): real(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdouble) */
    return 3.14;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackdouble) */
}


export passindouble_impl proc passindouble_impl( in _this: opaque, in d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passindouble) */
    return ( d == 3.14 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passindouble) */
}


export passoutdouble_impl proc passoutdouble_impl( in _this: opaque, out d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutdouble) */
    d = 3.14;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutdouble) */
}


export passinoutdouble_impl proc passinoutdouble_impl( in _this: opaque, inout d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdouble) */
    d = -d;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutdouble) */
}


export passeverywheredouble_impl proc passeverywheredouble_impl( in _this: opaque, in d1: real(64), out d2: real(64), inout d3: real(64)): real(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredouble) */
    d2 = 3.14;
    d3 = -d3;
    if ( d1 == 3.14 ) then return 3.14; else return 0.0 ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredouble) */
}


export returnbackfcomplex_impl proc returnbackfcomplex_impl( in _this: opaque): complex(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfcomplex) */
    return (3.1 + 3.1i):complex(64);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackfcomplex) */
}


export passinfcomplex_impl proc passinfcomplex_impl( in _this: opaque, in c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinfcomplex) */
    return ( c.re == 3.1 && c.im == 3.1 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinfcomplex) */
}


export passoutfcomplex_impl proc passoutfcomplex_impl( in _this: opaque, out c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutfcomplex) */
    c = (3.1 + 3.1i):complex(64);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutfcomplex) */
}


export passinoutfcomplex_impl proc passinoutfcomplex_impl( in _this: opaque, inout c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfcomplex) */
    c = (c.re,-c.im):complex(64);
    // was 'c=conj(c)', but I've had too many compilers complain... :[
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutfcomplex) */
}


export passeverywherefcomplex_impl proc passeverywherefcomplex_impl( in _this: opaque, in c1: complex(64), out c2: complex(64), inout c3: complex(64)): complex(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefcomplex) */
    var temp1: complex(64) = (3.1,3.1):complex(64);
    var error: complex(64) = (0.0, 0.0):complex(64);
    c2 = temp1;
    c3 = (c3.re,-c3.im):complex(64);
    // was 'c3=conj(c3)', but I've had too many compilers complain... :[
    if ( c1 == temp1 ) then return temp1; else return error ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefcomplex) */
}


export returnbackdcomplex_impl proc returnbackdcomplex_impl( in _this: opaque): complex(128) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdcomplex) */
    return (3.14, 3.14):complex(128);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackdcomplex) */
}


export passindcomplex_impl proc passindcomplex_impl( in _this: opaque, in c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passindcomplex) */
    return ( c.re == 3.14 && c.im == 3.14 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passindcomplex) */
}


export passoutdcomplex_impl proc passoutdcomplex_impl( in _this: opaque, out c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutdcomplex) */
    c = (3.14,3.14):complex(128);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutdcomplex) */
}


export passinoutdcomplex_impl proc passinoutdcomplex_impl( in _this: opaque, inout c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdcomplex) */
    c = (c.re, -c.im):complex(128);
    // was 'c=conj(c);' but I've had too many compilers complain for no reason.  :[
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutdcomplex) */
}


export passeverywheredcomplex_impl proc passeverywheredcomplex_impl( in _this: opaque, in c1: complex(128), out c2: complex(128), inout c3: complex(128)): complex(128) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredcomplex) */
    var temp : complex(128) = ( 3.14, 3.14 ):complex(128);
    var error: complex(128) = (0.0,0.0):complex(128);
    c2 = temp;
    c3 = (c3.re,-c3.im):complex(128); 
    // was 'c3=conj(c3);' but I've had too many compilers complain for no reason.  :[
    if ( c1 == temp ) then return temp; else return error;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredcomplex) */
}


/**
 * Implicit built-in method: _ctor
 */
export _ctor_impl proc _ctor_impl( in _this: opaque) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._ctor) */
    // Insert-Code-Here {Args.Basic._ctor} (constructor)
    /* DO-NOT-DELETE splicer.end(Args.Basic._ctor) */
}


/**
 * Implicit built-in method: _ctor2
 */
export _ctor2_impl proc _ctor2_impl( in _this: opaque, in private_data: opaque) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._ctor2) */
  // Insert-Code-Here {Args.Basic._ctor2} (ctor2)
    /* DO-NOT-DELETE splicer.end(Args.Basic._ctor2) */
}


/**
 * Implicit built-in method: _dtor
 */
export _dtor_impl proc _dtor_impl( in _this: opaque) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._dtor) */
    // Insert-Code-Here {Args.Basic._dtor} (destructor)
    /* DO-NOT-DELETE splicer.end(Args.Basic._dtor) */
}
;
