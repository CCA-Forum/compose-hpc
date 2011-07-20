

//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;
use sidl;
use blas;
use blas.VectorUtils_static;

type idxType = int(32);
type eltType = real(64);

config const rowSize = 20;
const colSize = rowSize;
config const blkSize = 4;
config const alpha = 2.0;
config const debug = false;

proc main() {

  writeln("Num Elements: ", ((rowSize: int(64)) * (colSize: int(64))));
  writeln("Block Size: ", blkSize);
  writeln("Alpha: ", alpha);

  const VectorDist = new BlockCyclic(startIdx=(1,1), blocksize=(1, blkSize));
  const VectorDom: domain(2, idxType) dmapped new dmap(VectorDist) = [1..rowSize, 1..colSize];
  
  var X: [VectorDom] eltType;
  var Y: [VectorDom] eltType;
  
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
  
  const startTime = getCurrentTime();
  cblas_daxpy_chpl(rowSize, alpha, X, Y);
  const endTime = getCurrentTime();
  
  if (debug) {
    write("2. Y: "); writeln(Y);
  }
  
  const execTime = endTime - startTime;
  
  writeln("Execution time = ", execTime, " secs");
  verifyResults(rowSize, alpha, X, Y);

}

proc cblas_daxpy_chpl(n, a, X, Y) {
	
  _extern proc double_ptr(inout firstElement: real(64)): opaque;
  
  forall r in 1..rowSize do {
   forall blk in 1..n by blkSize {
      on X(r, blk) do {
        if (debug) {
          writeln("Processing block: ", blk, " on locale-", here.id);	
        }
    	
        var xPtr = double_ptr(X(r, blk));
        var yPtr = double_ptr(Y(r, blk));
      
        var lower: [1..1] int(32); lower[1] = 0;
        var upper: [1..1] int(32); upper[1] = blkSize - 1;
        var stride: [1..1] int(32); stride[1] = 1;
      
        var xIor = sidl.sidl_double__array_borrow(xPtr, 1, lower[1], upper[1], stride[1]);      
        var xArr = new sidl.Array(real(64), sidl_double__array, xIor);
      
        var yIor = sidl_double__array_borrow(yPtr, 1, lower[1], upper[1], stride[1]); 
        var yArr = new sidl.Array(real(64), sidl_double__array, yIor);
       
        var baseEx: BaseException = nil;
        helper_daxpy(blkSize, a, xArr, yArr, baseEx);
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
