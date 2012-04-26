
use sidl;
class Args_Basic_Impl {
/* DO-NOT-DELETE splicer.begin(Args.Basic.Impl) */
  proc eps_compare(in a, in b):bool {
    var epsilon = 0.0005;
    return abs(a - b) < epsilon;
  }
/* DO-NOT-DELETE splicer.end(Args.Basic.Impl) */

/**
 * builtin method
 */
export Args_Basic__ctor_impl proc _ctor() {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._ctor) */
    // Insert-Code-Here {Args.Basic._ctor} (constructor)
    /* DO-NOT-DELETE splicer.end(Args.Basic._ctor) */
}


/**
 * builtin method
 */
export Args_Basic__ctor2_impl proc _ctor2(in private_data: opaque) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._ctor2) */
  // Insert-Code-Here {Args.Basic._ctor2} (ctor2)
    /* DO-NOT-DELETE splicer.end(Args.Basic._ctor2) */
}


/**
 * builtin method
 */
export Args_Basic__dtor_impl proc _dtor() {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._dtor) */
    // Insert-Code-Here {Args.Basic._dtor} (destructor)
    /* DO-NOT-DELETE splicer.end(Args.Basic._dtor) */
}


/**
 * builtin method
 */
export Args_Basic__load_impl proc _load() {
    /* DO-NOT-DELETE splicer.begin(Args.Basic._load) */
    /* DO-NOT-DELETE splicer.end(Args.Basic._load) */
}


export Args_Basic_returnbackbool_impl proc returnbackbool(): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackbool) */
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackbool) */
}


export Args_Basic_passinbool_impl proc passinbool(in b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinbool) */
    return b;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinbool) */
}


export Args_Basic_passoutbool_impl proc passoutbool(out b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutbool) */
    b = true;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutbool) */
}


export Args_Basic_passinoutbool_impl proc passinoutbool(inout b: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutbool) */
    b = !b;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutbool) */
}


export Args_Basic_passeverywherebool_impl proc passeverywherebool(in b1: bool, out b2: bool, inout b3: bool): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherebool) */
    b2 = true;
    b3 = !b3;
    return b1;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherebool) */
}


export Args_Basic_returnbackchar_impl proc returnbackchar(): string {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackchar) */
    return '3';
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackchar) */
}


export Args_Basic_passinchar_impl proc passinchar(in c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinchar) */
      return( c == '3');
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinchar) */
}


export Args_Basic_passoutchar_impl proc passoutchar(out c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutchar) */
    c = '3';
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutchar) */
}


export Args_Basic_passinoutchar_impl proc passinoutchar(inout c: string): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutchar) */
    if ( c >= 'a' && c <= 'z' ) { 
      //c += 'A' - 'a';
    } else if ( c >= 'A' && c <= 'Z' ) { 
      //c += 'a' - 'A';
    }
    return true;  
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutchar) */
}


export Args_Basic_passeverywherechar_impl proc passeverywherechar(in c1: string, out c2: string, inout c3: string): string {
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


export Args_Basic_returnbackint_impl proc returnbackint(): int(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackint) */
    return 3;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackint) */
}


export Args_Basic_passinint_impl proc passinint(in i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinint) */
    return ( i == 3 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinint) */
}


export Args_Basic_passoutint_impl proc passoutint(out i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutint) */
    i = 3;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutint) */
}


export Args_Basic_passinoutint_impl proc passinoutint(inout i: int(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutint) */
    i = -i;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutint) */
}


export Args_Basic_passeverywhereint_impl proc passeverywhereint(in i1: int(32), out i2: int(32), inout i3: int(32)): int(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywhereint) */
    i2 = 3;
    i3 = -i3;
    if ( i1 == 3 ) then 
      return 3 ;
    else return 0 ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywhereint) */
}


export Args_Basic_returnbacklong_impl proc returnbacklong(): int(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbacklong) */
    return 3;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbacklong) */
}


export Args_Basic_passinlong_impl proc passinlong(in l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinlong) */
    return ( l == 3 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinlong) */
}


export Args_Basic_passoutlong_impl proc passoutlong(out l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutlong) */
    l = 3;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutlong) */
}


export Args_Basic_passinoutlong_impl proc passinoutlong(inout l: int(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutlong) */
    l = -l;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutlong) */
}


export Args_Basic_passeverywherelong_impl proc passeverywherelong(in l1: int(64), out l2: int(64), inout l3: int(64)): int(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherelong) */
    l2 = 3;
    l3 = -l3;
    if ( l1 == 3 ) then return 3; else return 0;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherelong) */
}


export Args_Basic_returnbackfloat_impl proc returnbackfloat(): real(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfloat) */
    return 3.1:real(32);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackfloat) */
}


export Args_Basic_passinfloat_impl proc passinfloat(in f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinfloat) */
     return ( f == 3.1:real(32) );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinfloat) */
}


export Args_Basic_passoutfloat_impl proc passoutfloat(out f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutfloat) */
    f = 3.1:real(32);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutfloat) */
}


export Args_Basic_passinoutfloat_impl proc passinoutfloat(inout f: real(32)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfloat) */
    f = -f;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutfloat) */
}


export Args_Basic_passeverywherefloat_impl proc passeverywherefloat(in f1: real(32), out f2: real(32), inout f3: real(32)): real(32) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefloat) */
    f2 = 3.1:real(32);
    f3 = -f3;
    if ( f1 == 3.1:real(32) ) then return 3.1:real(32); else return 0.0:real(32);
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefloat) */
}


export Args_Basic_returnbackdouble_impl proc returnbackdouble(): real(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdouble) */
    return 3.14;
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackdouble) */
}


export Args_Basic_passindouble_impl proc passindouble(in d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passindouble) */
    return ( d == 3.14 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passindouble) */
}


export Args_Basic_passoutdouble_impl proc passoutdouble(out d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutdouble) */
    d = 3.14;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutdouble) */
}


export Args_Basic_passinoutdouble_impl proc passinoutdouble(inout d: real(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdouble) */
    d = -d;
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutdouble) */
}


export Args_Basic_passeverywheredouble_impl proc passeverywheredouble(in d1: real(64), out d2: real(64), inout d3: real(64)): real(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredouble) */
    d2 = 3.14;
    d3 = -d3;
    if ( d1 == 3.14 ) then return 3.14; else return 0.0 ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredouble) */
}


export Args_Basic_returnbackfcomplex_impl proc returnbackfcomplex(): complex(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackfcomplex) */
    return (3.1 + 3.1i):complex(64);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackfcomplex) */
}


export Args_Basic_passinfcomplex_impl proc passinfcomplex(in c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinfcomplex) */
  return ( eps_compare(c.re, 3.1) && eps_compare(c.im, 3.1) );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinfcomplex) */
}


export Args_Basic_passoutfcomplex_impl proc passoutfcomplex(out c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutfcomplex) */
    c = (3.1 + 3.1i):complex(64);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutfcomplex) */
}


export Args_Basic_passinoutfcomplex_impl proc passinoutfcomplex(inout c: complex(64)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutfcomplex) */
    c = (c.re,-c.im):complex(64);
    // was 'c=conj(c)', but I've had too many compilers complain... :[
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutfcomplex) */
}


export Args_Basic_passeverywherefcomplex_impl proc passeverywherefcomplex(in c1: complex(64), out c2: complex(64), inout c3: complex(64)): complex(64) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywherefcomplex) */
    var temp1: complex(64) = (3.1,3.1):complex(64);
    var error: complex(64) = (0.0, 0.0):complex(64);
    c2 = temp1;
    c3 = (c3.re,-c3.im):complex(64);
    // was 'c3=conj(c3)', but I've had too many compilers complain... :[
    if ( c1 == temp1 ) then return temp1; else return error ;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywherefcomplex) */
}


export Args_Basic_returnbackdcomplex_impl proc returnbackdcomplex(): complex(128) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.returnbackdcomplex) */
    return (3.14, 3.14):complex(128);
    /* DO-NOT-DELETE splicer.end(Args.Basic.returnbackdcomplex) */
}


export Args_Basic_passindcomplex_impl proc passindcomplex(in c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passindcomplex) */
    return ( c.re == 3.14 && c.im == 3.14 );
    /* DO-NOT-DELETE splicer.end(Args.Basic.passindcomplex) */
}


export Args_Basic_passoutdcomplex_impl proc passoutdcomplex(out c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passoutdcomplex) */
    c = (3.14,3.14):complex(128);
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passoutdcomplex) */
}


export Args_Basic_passinoutdcomplex_impl proc passinoutdcomplex(inout c: complex(128)): bool {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passinoutdcomplex) */
    c = (c.re, -c.im):complex(128);
    // was 'c=conj(c);' but I've had too many compilers complain for no reason.  :[
    return true;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passinoutdcomplex) */
}


export Args_Basic_passeverywheredcomplex_impl proc passeverywheredcomplex(in c1: complex(128), out c2: complex(128), inout c3: complex(128)): complex(128) {
    /* DO-NOT-DELETE splicer.begin(Args.Basic.passeverywheredcomplex) */
    var temp : complex(128) = ( 3.14, 3.14 ):complex(128);
    var error: complex(128) = (0.0,0.0):complex(128);
    c2 = temp;
    c3 = (c3.re,-c3.im):complex(128); 
    // was 'c3=conj(c3);' but I've had too many compilers complain for no reason.  :[
    if ( c1 == temp ) then return temp; else return error;
    /* DO-NOT-DELETE splicer.end(Args.Basic.passeverywheredcomplex) */
}

};
