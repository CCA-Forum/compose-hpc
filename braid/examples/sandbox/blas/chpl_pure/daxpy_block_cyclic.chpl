

//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;

type idxType = int(32);
type eltType = real(64);

config const rowSize = 20;
config const colSize = rowSize;
config const blkSize = 4;
config const alpha = 2.0;
config const debug = false;
config const mode = 2;

proc main() {

  writeln("Num Elements: ", ((rowSize: int(64)) * (colSize: int(64))));
  writeln("Block Size: ", blkSize);
  writeln("Alpha: ", alpha);

  const VectorDist = new BlockCyclic(startIdx=(1,1), blocksize=(1, blkSize));
  const VectorDom: domain(2, idxType) dmapped new dmap(VectorDist) = [1..rowSize, 1..colSize];
  
  var X: [VectorDom] eltType;
  var Y: [VectorDom] eltType;

  writeln("Initializing data...");
  
  forall r in 1..rowSize do {
    forall blk in 1..rowSize by blkSize {
      on Locales(X(r, blk).locale.id) do {
        forall i in [blk .. #blkSize] do {
          X(r, i) = r + i;
          Y(r, i) = 10 + r + i;
        }
      }
    } 
  }

  writeln("Done initalizing, executing...");
  
  const startTime = getCurrentTime();
  if (mode == 1) {
    chpl_daxpy_1(rowSize, alpha, X, Y);
  } else if (mode == 2) {
    chpl_daxpy_2(rowSize, alpha, X, Y);
  }
  const endTime = getCurrentTime();
  
  const execTime = endTime - startTime;
  
  writeln("Execution time = ", execTime, " secs");
  verifyResults(rowSize, alpha, X, Y);
 
}

proc chpl_daxpy_1(n, a, X, Y) {  
  Y = a * X + Y;
}

proc chpl_daxpy_2(n, a, X, Y) {
  forall r in 1..rowSize do {
    // writeln("r = ", r);
    forall blk in 1..n by blkSize {
      on X(r, blk) do {
        var rl = r;
        local { 
          [i in blk .. #blkSize] Y(rl, i) = a * X(rl, i) + Y(rl, i);
        }
      }
    }
  }
}

proc verifyResults(n, a, X, Y) {

  writeln("Verifying results...");

  var validMsg = "SUCCESS";
  forall r in 1..rowSize do {
    forall blk in 1..n by blkSize {
      on X(r, blk) do {
        const locDomain: domain(1) = [blk..#blkSize];
        forall i in locDomain do {
       	  var y_orig = 10 + r + i;
       	  var x = X(r, i);
       	  var y = Y(r, i);
        	
       	  var expected = y_orig + (a * x);
       	  if (abs(expected - y) > 0.0001) {
       	    validMsg = "FAILURE mismatch at index: " + i + ", expected: " + expected + ", found: " + y;
       	  }
        }
      }
    }
  }
  writeln("Validation: ", validMsg);
}



