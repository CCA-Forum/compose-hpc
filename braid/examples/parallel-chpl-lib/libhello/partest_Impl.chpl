
use sidl;
extern proc SET_TO_NULL(inout aRef);
// DO-NOT-DELETE splicer.begin(partest.Impl)
writeln("Unsolicited hi from", here.id);

coforall loc in Locales do
  if loc.id == 2 then
    on loc do
      writeln("Hello, world! ", "from node ", loc.id, " of ", numLocales);
// DO-NOT-DELETE splicer.end(partest.Impl)

class partest_Hello_Impl {
// DO-NOT-DELETE splicer.begin(partest.Hello.Impl)
// DO-NOT-DELETE splicer.end(partest.Hello.Impl)

/**
 * builtin method
 */
export partest_Hello__ctor_impl proc _ctor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(partest.Hello._ctor)
    // DO-NOT-DELETE splicer.end(partest.Hello._ctor)
}


/**
 * builtin method
 */
export partest_Hello__ctor2_impl proc _ctor2(in private_data: opaque, inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(partest.Hello._ctor2)
    // DO-NOT-DELETE splicer.end(partest.Hello._ctor2)
}


/**
 * builtin method
 */
export partest_Hello__dtor_impl proc _dtor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(partest.Hello._dtor)
    // DO-NOT-DELETE splicer.end(partest.Hello._dtor)
}


/**
 * builtin method
 */
export partest_Hello__load_impl proc _load(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(partest.Hello._load)
    // DO-NOT-DELETE splicer.end(partest.Hello._load)
}


export partest_Hello_sayHello_impl proc sayHello(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(partest.Hello.sayHello)
  writeln("Hello from locale #", here.id);
    // DO-NOT-DELETE splicer.end(partest.Hello.sayHello)
}

} // class partest_Hello_Impl


