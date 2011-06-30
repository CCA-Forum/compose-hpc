
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

// Start: Methods that will reside in the Skeleton file

proc createBlockDistArray2d_int_Skel( // _Impl
      lo1: int, hi1: int, 
      lo2: int, hi2: int, 
      blk1: int, blk2: int) {
  var myBlockedDomLiteral = 
    [lo1..hi1, lo2..hi2] dmapped Block([1..blk1, 1..blk2]);
  var myArray: [myBlockedDomLiteral] int(32);
  forall ba in myArray do
    ba = here.id;
  var distArray = new DistArray(myArray.eltType, myArray.rank, myArray);
  return distArray;
}

proc getFromDistArray2d_int_Skel(distArray, ind: int(32)... ?k) {
  return distArray.get((...ind));
}

proc setIntoDistArray2d_int_Skel(distArray, newVal: int(32), ind: int(32)... ?k) {
  distArray.set(newVal, (...ind));
}

// End: Methods that will reside in the Skeleton file

///////////////////////////////////////////

// Start: Prepare main

proc main_dummy_calls() {
  var myDistArr = createBlockDistArray2d_int_Skel(1, 4, 1, 6, 2, 3);
  setIntoDistArray2d_int_Skel(myDistArr, 42, 2, 2);
  var res = getFromDistArray2d_int_Skel(myDistArr, 2, 2);
}

_extern proc main_server();

proc main() {

  main_dummy_calls();  

  main_server();

}

// End: Prepare main

///////////////////////////////////////////
// 
// Start: example usage
// 
// writeln("Starting Int Dist Array...");
// 
// var myBlockDist: dmap(Block(rank=2)) = new dmap(new Block([1..5, 1..6]));
// type myBlockedDomType = domain(2) dmapped myBlockDist;
// var myBlockedDomLiteral = [1..4,1..6] dmapped Block([1..2,1..3]);
// var myArray: [myBlockedDomLiteral] int(32);
// 
// writeln("myArray:");
// writeln("--------");
// writeln("myArray.type: ", typeToString(myArray.type));
// writeln(myArray);
// 
// var myDistArr = createDistArray(myArray);
// 
// writeln("1. myDistArr.get(2, 2): ", myDistArr.get(2, 2));
// myDistArr.set(22, 2, 2);
// writeln("2. myDistArr.get(2, 2): ", myDistArr.get(2, 2));
// 
// setIntoDistArray(myDistArr, 42, 2, 2);
// writeln("3. myDistArr.get(2, 2): ", getFromDistArray(myDistArr, 2, 2));
// 
// 
// writeln("Ending Int Dist Array.");
// 
// End: example usage
// 
///////////////////////////////////////////




