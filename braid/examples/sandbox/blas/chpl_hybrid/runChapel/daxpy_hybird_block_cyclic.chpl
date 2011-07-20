

//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;
use sidl;
use blas;
use blas.VectorUtils_static;

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
      var elemLocId = X(blk).locale.id;
      forall i in [blk .. #blkSize] do {
        X(i) = i;
        Y(i) = 10 + i;
      }
    }
  }
  
  if (debug) {
    writeln("1. alpha: ", alpha);
    write("1. X: "); writeln(X);
    write("1. Y: "); writeln(Y);
  }
  
  const startTime = getCurrentTime();
  cblas_daxpy_chpl(numElements, alpha, X, Y);
  const endTime = getCurrentTime();
  
  if (debug) {
    write("2. Y: "); writeln(Y);
  }
  
  const execTime = endTime - startTime;
  
  writeln("Execution time = ", execTime, " secs");
  verifyResults(numElements, alpha, X, Y);

}

proc cblas_daxpy_chpl(n, a, X, Y) {
	
  // _extern record sidl__array { };	
  // _extern class sidl_double__array { var d_metadata: sidl__array; var d_firstElement: opaque; };	
  _extern proc double_ptr(inout firstElement: real(64)): opaque;
  
  forall blk in 1..n by blkSize {
    on X(blk) do {
      if (debug) {
        writeln("Processing block: ", blk, " on locale-", here.id);	
      }
    	
      var xPtr = double_ptr(X(blk));
      var yPtr = double_ptr(Y(blk));
      
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
