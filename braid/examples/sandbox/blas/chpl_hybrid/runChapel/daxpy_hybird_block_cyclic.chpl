

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
  
  if (debug) {
    writeln("1. alpha: ", alpha);
    write("1. X: "); writeln(X);
    write("1. Y: "); writeln(Y);
  }
  
  const startTime = getCurrentTime();
  cblas_daxpy(numElements, alpha, X, Y);
  const endTime = getCurrentTime();
  
  if (debug) {
    write("2. Y: "); writeln(Y);
  }
  
  const execTime = endTime - startTime;
  
  writeln("Execution time = ", execTime, " secs");
  verifyResults(numElements, alpha, X, Y);

}

proc cblas_daxpy(n, a, X, Y) {
	
  // _extern record sidl__array { };	
  // _extern class sidl_double__array { var d_metadata: sidl__array; var d_firstElement: opaque; };	
  _extern proc double_ptr(inout firstElement: real(64)): opaque;
  
  for blk in 1..n by blkSize {
    on Locales(X(blk).locale.id) do {
    
      writeln("Processing block: ", blk, " on locale-", here.id);	
    	
      var xPtr = double_ptr(X(blk));
      var yPtr = double_ptr(Y(blk));
      
      var lower: [1..1] int(32); lower[1] = 0;
      var upper: [1..1] int(32); upper[1] = blkSize - 1;
      var stride: [1..1] int(32); stride[1] = 1;
      writeln(" lower: ", lower, ", upper: ", upper, ", stride: ", stride);
      
      var xIor = sidl.sidl_double__array_borrow(xPtr, 1, lower[1], upper[1], stride[1]);      
      var xArr = new sidl.Array(real(64), sidl_double__array, xIor);
      
      var yIor = sidl_double__array_borrow(yPtr, 1, lower[1], upper[1], stride[1]); 
      var yArr = new sidl.Array(real(64), sidl_double__array, yIor);
       
      writeln(" calling helper_daxpy");
      helper_daxpy(n, a, xArr, yArr);
    }
  }
}

proc verifyResults(n, a, X, Y) {

  writeln("Verifying results...");
  var validMsg = "SUCCESS";
  forall i in X.domain do {
    var yl = Y(i).locale.id;
    
    var x = X(i);
    var y = Y(i);
    
    var d = y - (a * x);
    
    if (abs(d - yl) > 0.0001) {
      validMsg = "FAILURE";
    }
  }
  writeln("Validation: ", validMsg);
}


