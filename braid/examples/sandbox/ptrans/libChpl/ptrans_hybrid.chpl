//  ===============================================
//  Chapel Implementation of HPCC PTRANS Benchmark
//  Compute  C = beta C + A', where  A  and  C  are 
//  large distributed dense matrices.
//  Original version by John G. Lewis, Dec 2010
//  ===============================================

// command: ./a.out -nl 4 --numrows=20 --rowBlkSize=3 --colBlkSize=5 --printStats=false --beta=0.5


//
// Use standard Chapel modules for Block-Cyclic distributions and timings
//
use BlockCycDist, Time;


//
// Use shared user module for computing HPCC problem sizes
//
use HPCCProblemSize;

use ptrans_hybrid_support;
use hplsupport;
use hpcc.ParallelTranspose_static;

//
// The number of arrays (used to compute the default problem size)
//
const numMatrices = 2;

//
// Types used to represent matrix indices and elements (overridable on
// compiler command line)
//
config type idxType = int(32);
config type eltType = real(64);

//
// Configuration constants to set the problem size (numrows x numcols),
// the block sizes, and the scalar multiplier, beta.
//
config const numrows = computeProblemSize(numMatrices, eltType, rank=2, 
                                          retType=idxType), 
             numcols = numrows,
             rowBlkSize = 8, 
             colBlkSize = rowBlkSize,
             beta = 1.0;

//
// Configuration constant used for verification thresholds
//
config const epsilon = 2.2e-16;

//
// Configuration constants to control what's printed -- benchmark
// parameters, input and output arrays, and/or statistics
//
config const printParams = true,
             printArrays = false,
             printStats = true;

// FIXME: Will a ref always be an int(32) in chpl?
_extern proc GET_CHPL_REF(inData): int(32);

//
// Start here
//
proc main() {
	
  ptrans_support_dummy_calls();	
	
  printConfiguration();

  //
  // Create Block-Cyclic distributions for both the Matrix and its
  // transpose:
  //
  const MatrixDist = new BlockCyclic(startIdx=(1,1),
                                     blocksize=(rowBlkSize, colBlkSize));

  const TransposeDist = new BlockCyclic(startIdx=(1,1),
                                        blocksize=(colBlkSize, rowBlkSize));

  //
  // Declare domains (index sets) for the Matrix and its transpose
  // using the distributions above:
  //
  const MatrixDom     : domain(2, idxType) dmapped new dmap(MatrixDist) 
                      = [1..numrows, 1..numcols],

        TransposeDom  : domain(2, idxType) dmapped new dmap(TransposeDist) 
                       = [1..numcols, 1..numrows];

  //
  // Declare the matrices themselves
  //
  var A                  : [MatrixDom   ] eltType, 
      C                  : [TransposeDom] eltType;

  const error_tolerance = initArrays(A, C);

  // ------------------------
  // Compute  C = beta C + A'
  // ------------------------

  const startTime = getCurrentTime();
    
  // writeln("1a. A: "); writeln(A);
  // writeln("1b. C: "); writeln(C);
		
  forall (i,j) in TransposeDom do {
    var aWrapper = new hplsupport.BlockCyclicDistArray2dDouble();	
    aWrapper.initData(GET_CHPL_REF(A));
    
    var cWrapper = new hplsupport.BlockCyclicDistArray2dDouble();	  
    cWrapper.initData(GET_CHPL_REF(C));  	  
	  
    // C[i,j] = beta * C[i,j]  +  A[j,i];
    ptransCompute(aWrapper, cWrapper, beta, i, j);
  }
  
  const execTime = getCurrentTime() - startTime;
  
  const validAnswer = verifyResults(C, error_tolerance);
  printResults(validAnswer, execTime);
}


//
// print out the problem size and block size if requested
//
proc printConfiguration() {
  if (printParams) {
    if (printStats) then printLocalesTasks();
    printProblemSize(eltType, numMatrices, numrows, rank=2, numcols);
    writeln("block size = ", rowBlkSize, " x ", colBlkSize);
    writeln("beta = ", beta);
    writeln();
  }
}

//
// Generate test matrices and explicit comparative result.  The real
// HPCC benchmark uses a fragmented memory model random number.
// We substitute less expensive test matrices that are still likely
// to detect any addressing errors.
//
proc initArrays(A, C) {
  forall (i,j) in A.domain do
    A[i,j] = erf(i) * cos(j);

  forall (i,j) in C.domain do
    C[i,j] = sin(j) * cbrt(i);

  const norm_A = sqrt( + reduce A**2 ),
        norm_C = sqrt( + reduce C**2 );

  const error_tolerance = (norm_A  +  abs(beta) * norm_C) * epsilon;

  if (printArrays) {
    writeln("A is:\n", A);
    writeln("C (initially) is:\n", C);
    writeln("norm of A: ", norm_A);
    writeln("norm of C: ", norm_C);
  }

  if (printStats) then writeln("max acceptable discrepancy: ", error_tolerance);

  return error_tolerance;
}


//
// Helper function to compute what the expected result is based on
// the input array values
//
proc CPlusATranspose((i,j)) {
  return beta * sin(j) * cbrt(i) + erf(j) * cos(i);
}

                            
//
// Verify that the computation was correct
//
proc verifyResults(C: [], tolerance) {
  if (printArrays) then writeln("C (result) is:\n", C);

  const error = max reduce [ij in C.domain] abs(C(ij) - CPlusATranspose(ij));

  if (printStats) then writeln("error = ", error);

  return (error <= tolerance);
}


//
// Print out success or failure, timing, and GB/sec
//
proc printResults(successful, execTime) { 
  writeln("Validation: ", if successful then "SUCCESS" else "FAILURE");

  const GBPerSec = if (execTime  > 0.0) then
                     (numrows*numcols*numBytes(eltType)) / (1.0e9 * execTime)
                   else
                     0.0;

  if printStats {
    writeln("Execution time = ", execTime);
    writeln("Performance (GB/sec) = ", GBPerSec);
  }
}
