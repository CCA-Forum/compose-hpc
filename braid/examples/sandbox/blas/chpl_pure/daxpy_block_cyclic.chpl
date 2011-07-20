

//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;

type idxType = int(32);
type eltType = real(64);

config const numElements = 20;
config const blkSize = 4;
config const alpha = 2.0;
config const debug = false;

proc main() {

  writeln("Num Elements: ", numElements);
  writeln("Block Size: ", blkSize);
  writeln("Alpha: ", alpha);

  var startIndicesTuple: 1*idxType;
  startIndicesTuple(1) = 1;
  
  var blockSizeTuple: 1*idxType;
  blockSizeTuple(1) = blkSize;
  
  const VectorDist = new BlockCyclic(startIdx=startIndicesTuple, blocksize=blockSizeTuple);
  const VectorDom: domain(1, idxType) dmapped new dmap(VectorDist) = [1..numElements];
  
  var X: [VectorDom] eltType;
  var Y: [VectorDom] eltType;
  
  forall blk in 1..numElements by blkSize {
    on Locales(X(blk).locale.id) do {

      forall i in [blk .. #blkSize] do {
        X(i) = i;
        Y(i) = 10 + i;
      }
    }
  }
  writeln("Initialized data");  
  if (debug) {
    writeln("1. alpha: ", alpha);
    write("1. X: "); writeln(X);
    write("1. Y: "); writeln(Y);
  }
  
  const startTime = getCurrentTime();
  chpl_daxpy(numElements, alpha, X, Y);
  const endTime = getCurrentTime();
  
  if (debug) {
    write("2. Y: "); writeln(Y);
  }
  
  const execTime = endTime - startTime;
  
  writeln("Execution time = ", execTime, " secs");
  verifyResults(numElements, alpha, X, Y);

  const startTime2 = getCurrentTime();
  chpl_daxpy_2(numElements, alpha, X, Y);
  const endTime2 = getCurrentTime();
  writeln("Variant-2 time = ", (endTime2 - startTime2), " secs");  
  
  const startTime3 = getCurrentTime();
  chpl_daxpy_3(numElements, alpha, X, Y);
  const endTime3 = getCurrentTime();
  writeln("Variant-3 time = ", (endTime3 - startTime3), " secs");  
 
}

proc chpl_daxpy(n, a, X, Y) {  
  forall blk in 1..n by blkSize {
    on X(blk) do {
      const locDomain: domain(1) = [blk..#blkSize];
      forall i in locDomain do {
        Y(i) = (a * X(i)) + Y(i);
      }
    }
  }
}

proc chpl_daxpy_2(n, a, X, Y) {  
  Y = a * X + Y;
}

proc chpl_daxpy_3(n, a, X, Y) {  
  forall blk in 1..n by blkSize {
    on X(blk) do {
      const locDomain: domain(1) = [blk..#blkSize];
      forall (x,y) in (X(locDomain), Y(locDomain)) do {
    	  y = a * x + y;
      }
    }
  }	 
}

proc verifyResults(n, a, X, Y) {

  writeln("Verifying results...");

  var validMsg = "SUCCESS";
  forall blk in 1..n by blkSize {
    on X(blk) do {
      const locDomain: domain(1) = [blk..#blkSize];
      forall i in locDomain do {
      	var y_orig = 10 + i;
       	var x = X(i);
       	var y = Y(i);
        	
       	var expected = y_orig + (a * x);
       	if (abs(expected - y) > 0.0001) {
       	  validMsg = "FAILURE mismatch at index: " + i + ", expected: " + expected + ", found: " + y;
       	}
      }
    }
  }
  writeln("Validation: ", validMsg);
}



