
use sidl;
extern proc SET_TO_NULL(inout aRef);
// DO-NOT-DELETE splicer.begin(pgas.Impl)

//use BlockDist;

// DO-NOT-DELETE splicer.end(pgas.Impl)

class pgas_blockedDouble3dArray_Impl {
// DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray.Impl)

  // tiny default allocation
  const CubeIndex = [0..0, 0..0, 0..0];
  var CubeDom = [CubeIndex] dmapped Block(CubeIndex);
  var data: [CubeDom] real(64);

// DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray.Impl)

/**
 * builtin method
 */
export pgas_blockedDouble3dArray__ctor_impl proc _ctor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray._ctor)
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray._ctor)
}


/**
 * builtin method
 */
export pgas_blockedDouble3dArray__ctor2_impl proc _ctor2(in private_data: opaque, inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray._ctor2)
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray._ctor2)
}


/**
 * builtin method
 */
export pgas_blockedDouble3dArray__dtor_impl proc _dtor(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray._dtor)
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray._dtor)
}


/**
 * builtin method
 */
export pgas_blockedDouble3dArray__load_impl proc _load(inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray._load)
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray._load)
}


/**
 * 
 * allocate a blocked cubic array of doubles in size*size*size
 * 
 */
export pgas_blockedDouble3dArray_allocate_impl proc allocate(in size: int(32), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray.allocate)

  // dynamically resize data
  const CubeIndex = [0..#size, 0..#size, 0..#size];
  //CubeDom = [CubeIndex] dmapped Block(CubeIndex);
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray.allocate)
}


export pgas_blockedDouble3dArray_get_impl proc get(in idx1: int(32), in idx2: int(32), in idx3: int(32), inout _ex: sidl.sidl_BaseInterface__object): real(64) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray.get)
    //data[idx1, idx2, idx3];
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray.get)
}


export pgas_blockedDouble3dArray_set_impl proc set(in idx1: int(32), in idx2: int(32), in idx3: int(32), in val: real(64), inout _ex: sidl.sidl_BaseInterface__object) {
    SET_TO_NULL(_ex);
    // DO-NOT-DELETE splicer.begin(pgas.blockedDouble3dArray.set)
    //data[idx1, idx2, idx3] = val;
    // DO-NOT-DELETE splicer.end(pgas.blockedDouble3dArray.set)
}

} // class pgas_blockedDouble3dArray_Impl


