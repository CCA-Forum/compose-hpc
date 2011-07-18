

//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;

type idxType = int(32);
type eltType = real(64);

config const numElements = 20;
config const blkSize = 4;
config const alpha = 2.0;

proc main() {

  var startIndicesTuple: 1*idxType;
  startIndicesTuple(1) = 1;
  
  var blockSizeTuple: 1*idxType;
  blockSizeTuple(1) = blkSize;
  
  const VectorDist = new BlockCyclic(startIdx=startIndicesTuple, blocksize=blockSizeTuple);
  const VectorDom: domain(1, idxType) dmapped new dmap(VectorDist) = [1..numElements];
  
  var X: [VectorDom] eltType;
  var Y: [VectorDom] eltType;
  
  forall i in VectorDom do {
    var elemLocId = X(i).locale.id;
    X(i) = i;
    Y(i) = elemLocId;
  }
  
  writeln("1. alpha: ", alpha);
  write("1. X: "); writeln(X);
  write("1. Y: "); writeln(Y);
  
  cblas_daxpy(numElements, alpha, X, Y);
  
  write("2. Y: "); writeln(Y);

}

proc cblas_daxpy(n, a, X, Y) {
  
  for blk in 1..n by blkSize {
    on Locales(X(blk).locale.id) do {
      const locDomain:domain(1) = [blk..#blkSize];
      for i in locDomain do {
        Y(i) = (a * X(i)) + Y(i);
      }
    }
  }
}


