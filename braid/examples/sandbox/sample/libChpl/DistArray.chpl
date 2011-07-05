
use BlockDist;

class DistArray {

  type eltType;
  param rank;
  var distArr;

  proc get(ind: rank * int(32)): eltType {
    return distArr(ind);
  }

  proc get(ind: int(32)... ?k): eltType where k == rank {
    return distArr((...ind));
  }

  proc set(newVal: eltType, ind: rank * int(32)) {
    distArr(ind) = newVal;
  }

  proc set(newVal: eltType, ind: int(32)... ?k) where k == rank {
    distArr((...ind)) = newVal;
  }
}

///////////////////////////////////////////

