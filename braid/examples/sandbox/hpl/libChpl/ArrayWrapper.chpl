class ArrayWrapper {

  type eltType;
  param rank;
  var wrappedArray;

  proc get(ind: rank * int(32)): eltType {
    return wrappedArray(ind);
  }

  proc get(ind: int(32)... ?k): eltType where k == rank {
    return wrappedArray((...ind));
  }

  proc set(newVal: eltType, ind: rank * int(32)) {
    wrappedArray(ind) = newVal;
  }

  proc set(newVal: eltType, ind: int(32)... ?k) where k == rank {
    wrappedArray((...ind)) = newVal;
  }
}