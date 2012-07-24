
use sidl;
extern proc SET_TO_NULL(inout aRef);
// DO-NOT-DELETE splicer.begin(pgas.Impl)
writeln("Unsolicited hello from locale ", here.id);
coforall loc in Locales do
  if loc.id == 2 then
    on loc do
      writeln("Hello, world! ", "from node ", loc.id, " of ", numLocales);

use BlockDist;
// tiny default allocation
const CubeIndex = [0..0, 0..0, 0..0];
var CubeDom = [CubeIndex] dmapped Block(CubeIndex);
var x: [CubeDom] real(64);
var y: [CubeDom] real(64);
var z: [CubeDom] real(64);
var xd: [CubeDom] real(64);
var yd: [CubeDom] real(64);
var zd: [CubeDom] real(64);

// DO-NOT-DELETE splicer.end(pgas.Impl)

class pgas_GlobalData_Impl {
// DO-NOT-DELETE splicer.begin(pgas.GlobalData.Impl)
// DO-NOT-DELETE splicer.end(pgas.GlobalData.Impl)

/**
 * builtin method
 */
export pgas_GlobalData__ctor_impl proc _ctor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData._ctor)
    //    writeln("In ctor / locale ", here.id);
    // DO-NOT-DELETE splicer.end(pgas.GlobalData._ctor)
}


/**
 * builtin method
 */
export pgas_GlobalData__ctor2_impl proc _ctor2(in private_data: opaque, inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData._ctor2)
    // DO-NOT-DELETE splicer.end(pgas.GlobalData._ctor2)
}


/**
 * builtin method
 */
export pgas_GlobalData__dtor_impl proc _dtor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData._dtor)
    // DO-NOT-DELETE splicer.end(pgas.GlobalData._dtor)
}


/**
 * builtin method
 */
export pgas_GlobalData__load_impl proc _load(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData._load)
    // DO-NOT-DELETE splicer.end(pgas.GlobalData._load)
}


/**
 * 
 * allocate a blocked cubic array of doubles in size*size*size
 * 
 */
export pgas_GlobalData_allocate_impl proc allocate(in size: int(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.allocate)
    //var lock$: sync bool;
    if here.id == 0 then {
      writeln("allocating 3x",size);
      // dynamically resize data
      const CubeIndex = [0..#size, 0..#size, 0..#size];
      CubeDom = [CubeIndex] dmapped Block(CubeIndex);
      //lock$ = false;
    }
    //var unlock = lock$;
    /* coforall loc in Locales do */
    /*   on loc do { */
    /* 	writeln("access on ", here.id, " = ", loc.id); */
    /* 	var i = loc.id; */
    /* 	x[i,i,i] = 0; */
    /* 	y[i,i,i] = 0; */
    /* 	z[i,i,i] = 0; */
    /* 	xd[i,i,i] = 0; */
    /* 	yd[i,i,i] = 0; */
    /* 	zd[i,i,i] = 0; */
    /*   } */
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.allocate)
}


export pgas_GlobalData_getx_impl proc getx(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.getx)
    return x[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.getx)
}


export pgas_GlobalData_setx_impl proc setx(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.setx)
    x[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.setx)
}


export pgas_GlobalData_gety_impl proc gety(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.gety)
    return y[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.gety)
}


export pgas_GlobalData_sety_impl proc sety(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.sety)
    y[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.sety)
}


export pgas_GlobalData_getz_impl proc getz(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.getz)
    return z[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.getz)
}


export pgas_GlobalData_setz_impl proc setz(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.setz)
    z[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.setz)
}


export pgas_GlobalData_getxd_impl proc getxd(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.getxd)
    return xd[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.getxd)
}


export pgas_GlobalData_setxd_impl proc setxd(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.setxd)
    xd[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.setxd)
}


export pgas_GlobalData_getyd_impl proc getyd(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.getyd)
    return yd[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.getyd)
}


export pgas_GlobalData_setyd_impl proc setyd(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.setyd)
    yd[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.setyd)
}


export pgas_GlobalData_getzd_impl proc getzd(in idx1: int(64), in idx2: int(64), in idx3: int(64), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.getzd)
    return zd[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.getzd)
}


export pgas_GlobalData_setzd_impl proc setzd(in idx1: int(64), in idx2: int(64), in idx3: int(64), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.GlobalData.setzd)
    zd[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.GlobalData.setzd)
}

} // class pgas_GlobalData_Impl


