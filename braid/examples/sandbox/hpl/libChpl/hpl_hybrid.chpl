// obtained from CHPL_HOME/examples/hpcc/hpl.chpl

//
// Use standard modules for vector and matrix Norms, Random numbers
// and Timing routines
//
use Norm, Random, Time;

//
// Use the user module for computing HPCC problem sizes
//
use HPCCProblemSize;

//
// Use the distributions we need for this computation
//
use BlockCycDist;

use ArrayWrapper;
use hplsupport_BlockCyclicDistArray2dDouble_chplImpl;
use hplsupport_SimpleArray1dInt_chplImpl;


_extern proc panelSolveNative(inout abData, inout pivData, 
		/* abLimits*/ in abStart1: int(32), in abEnd1: int(32), in abStart2: int(32), in abEnd2: int(32), 
		/*panel domain*/ in start1: int(32), in end1: int(32), in start2: int(32), in end2: int(32));


//
// The number of matrices and the element type of those matrices
// indexType can be int or int(64), elemType can be real or complex
//
const numMatrices = 1;
type indexType = int(32),
     elemType = real(64);

//
// Configuration constants indicating the problem size (n) and the
// block size (blkSize)
//
config const n = computeProblemSize(numMatrices, elemType, rank=2, 
                                    memFraction=2, retType=indexType),
             blkSize = 8;

//
// Configuration constant used for verification thresholds
//
config const epsilon = 2.0e-15;

//
// Configuration constants to indicate whether or not to use a
// pseudo-random seed (based on the clock) or a fixed seed; and to
// specify the fixed seed explicitly
//
config const useRandomSeed = true,
             seed = if useRandomSeed then SeedGenerator.currentTime else 31415;

//
// Configuration constants to control what's printed -- benchmark
// parameters, input and output arrays, and/or statistics
//
config const printParams = true,
             printArrays = false,
             printStats = true;

//
// The program entry point
//
proc main() {
  dummy_calls();
	
  printConfiguration();

  //
  // MatVectSpace is a 2D domain of type indexType that represents the
  // n x n matrix adjacent to the column vector b.  MatrixSpace is a
  // subdomain that is created by slicing into MatVectSpace,
  // inheriting all of its rows and its low column bound.  As our
  // standard distribution library is filled out, MatVectSpace will be
  // distributed using a BlockCyclic(blkSize) distribution.
  //
  const MatVectSpace: domain(2, indexType) 
                      dmapped BlockCyclic(startIdx=(1,1), (blkSize,blkSize)) 
                    = [1..n, 1..n+1],
        MatrixSpace = MatVectSpace[.., ..n];

  var Ab : [MatVectSpace] elemType,  // the matrix A and vector b
      piv: [1..n] indexType;         // a vector of pivot values

  initAB(Ab);

  const startTime = getCurrentTime();     // capture the start time

  LUFactorize(n, Ab, piv);                 // compute the LU factorization

  var x = backwardSub(n, Ab);  // perform the back substitution

  const execTime = getCurrentTime() - startTime;  // store the elapsed time

  //
  // Validate the answer and print the results
  const validAnswer = verifyResults(Ab, MatrixSpace, x);
  printResults(validAnswer, execTime);
}

//
// blocked LU factorization with pivoting for matrix augmented with
// vector of RHS values.
//
proc LUFactorize(n: indexType, Ab: [?AbD] elemType,
                piv: [1..n] indexType) {
  
  // Initialize the pivot vector to represent the initially unpivoted matrix.
  piv = 1..n;

  /* The following diagram illustrates how we partition the matrix.
     Each iteration of the loop increments a variable blk by blkSize;
     point (blk, blk) is the upper-left location of the currently
     unfactored matrix (the dotted region represents the areas
     factored in prior iterations).  The unfactored matrix is
     partioned into four subdomains: tl, tr, bl, and br, and an
     additional domain (not shown), l, that is the union of tl and bl.

                (point blk, blk)
       +-------//------------------+
       |......//...................|
       |.....//....................|
       |....+-----+----------------|
       |....|     |                |
       |....| tl  |      tr        |
       |....|     |                |
       |....+-----+----------------|
       |....|     |                |
       |....|     |                |
       |....| bl  |      br        |
       |....|     |                |
       |....|     |                |
       +----+-----+----------------+
  */
  for blk in 1..n by blkSize {
    const tl = AbD[blk..#blkSize, blk..#blkSize],
          tr = AbD[blk..#blkSize, blk+blkSize..],
          bl = AbD[blk+blkSize.., blk..#blkSize],
          br = AbD[blk+blkSize.., blk+blkSize..],
          l  = AbD[blk.., blk..#blkSize];

    //
    // Now that we've sliced and diced Ab properly, do the blocked-LU
    // computation:
    //
    panelSolve(Ab, l, piv);
    updateBlockRow(Ab, tl, tr);
    
    //
    // update trailing submatrix (if any)
    //
    schurComplement(Ab, bl, tr, br);
  }
}

//
// Distributed matrix-multiply for HPL. The idea behind this algorithm is that
// some point the matrix will be partioned as shown in the following diagram:
//
//    [1]----+-----+-----+-----+
//     |     |bbbbb|bbbbb|bbbbb|  Solve for the dotted region by
//     |     |bbbbb|bbbbb|bbbbb|  multiplying the 'a' and 'b' region.
//     |     |bbbbb|bbbbb|bbbbb|  The 'a' region is a block column, the
//     +----[2]----+-----+-----+  'b' region is a block row.
//     |aaaaa|.....|.....|.....|
//     |aaaaa|.....|.....|.....|  The 'a' region was 'bl' in the calling
//     |aaaaa|.....|.....|.....|  function but called AD here.  Similarly,
//     +-----+-----+-----+-----+  'b' was 'tr' in the calling code, but BD
//     |aaaaa|.....|.....|.....|  here.
//     |aaaaa|.....|.....|.....|  
//     |aaaaa|.....|.....|.....|
//     +-----+-----+-----+-----+
//
// Every locale with a block of data in the dotted region updates
// itself by multiplying the neighboring a-region block to its left
// with the neighboring b-region block above it and subtracting its
// current data from the result of this multiplication. To ensure that
// all locales have local copies of the data needed to perform this
// multiplication we copy the data A and B data into the replA and
// replB arrays, which will use a dimensional (block-cyclic,
// replicated-block) distribution (or vice-versa) to ensure that every
// locale only stores one copy of each block it requires for all of
// its rows/columns.
//
proc schurComplement(Ab: [?AbD] elemType, AD: domain, BD: domain, Rest: domain) {
  //
  // Copy data into replicated array so every processor has a local copy
  // of the data it will need to perform a local matrix-multiply.  These
  // replicated distributions aren't implemented yet, but imagine that
  // they look something like the following:
  //
  //var replAbD: domain(2) 
  //            dmapped new Dimensional(BlkCyc(blkSize), Replicated)) = AbD[AD];
  //
  const replAD: domain(2, indexType) = AD,
        replBD: domain(2, indexType) = BD;
    
  const replA : [replAD] elemType = Ab[replAD],
        replB : [replBD] elemType = Ab[replBD];

  //  writeln("Rest = ", Rest);
  //  writeln("Rest by blkSize = ", Rest by (blkSize, blkSize));
  // do local matrix-multiply on a block-by-block basis
  forall (row,col) in Rest by (blkSize, blkSize) {
    //
    // At this point, the dgemms should all be local once we have
    // replication correct, so we'll want to assert that fact
    //
    //    local {
      const aBlkD = replAD[row..#blkSize, ..],
            bBlkD = replBD[.., col..#blkSize],
            cBlkD = AbD[row..#blkSize, col..#blkSize];

      dgemmNativeInds(replA[aBlkD], replB[bBlkD], Ab[cBlkD]);
      //    }
  }
}

//
// calculate C = C - A * B.
//
proc dgemmNativeInds(A: [] elemType,
                    B: [] elemType,
                    C: [] elemType) {
  for (iA, iC) in (A.domain.dim(1), C.domain.dim(1)) do
    for (jA, iB) in (A.domain.dim(2), B.domain.dim(1)) do
      for (jB, jC) in (B.domain.dim(2), C.domain.dim(2)) do
        C[iC,jC] -= A[iA, jA] * B[iB, jB];
}



//
// do unblocked-LU decomposition within the specified panel, update the
// pivot vector accordingly
//
proc panelSolve(ab: [] elemType,
               panel: domain,
               piv: [] indexType) {

  var abWrapper = new ArrayWrapper(elemType, 2, ab);
  var pivWrapper = new ArrayWrapper(indexType, 1, piv);	
  
  var abDom = ab.domain;
  panelSolveNative(abWrapper, pivWrapper, 
  		  abDom.low(1), abDom.high(1), abDom.low(2), abDom.high(2), 
  		  panel.low(1), panel.high(1), panel.low(2), panel.high(2));
}

//
// Update the block row (tr for top-right) portion of the matrix in a
// blocked LU decomposition.  Each step of the LU decomposition will
// solve a block (tl for top-left) portion of a matrix. This function
// solves the rows to the right of the block.
//
proc updateBlockRow(Ab: [] elemType,
                   tl: domain,
                   tr: domain) {

  for row in tr.dim(1) {
    const activeRow = tr[row..row, ..],
          prevRows = tr.dim(1).low..row-1;

    forall (i,j) in activeRow do
      for k in prevRows do
        Ab[i, j] -= Ab[i, k] * Ab[k,j];
  }
}


//
// compute the backwards substitution
//
proc backwardSub(n: indexType,
                 Ab: [] elemType) {
  const bd = Ab.domain.dim(1);
  var x: [bd] elemType;

  for i in bd by -1 do
    x[i] = (Ab[n+1,i] - (+ reduce [j in i+1..bd.high] (Ab[i,j] * x[j]))) 
            / Ab[i,i];

  return x;
}

//
// print out the problem size and block size if requested
//
proc printConfiguration() {
  if (printParams) {
    if (printStats) then printLocalesTasks();
    printProblemSize(elemType, numMatrices, n, rank=2);
    writeln("block size = ", blkSize, "\n");
  }
}

//   
// construct an n by n+1 matrix filled with random values and scale
// it to be in the range -1.0..1.0
//
proc initAB(Ab: [] elemType) {
  fillRandom(Ab, seed);
  Ab = Ab * 2.0 - 1.0;
}

//
// calculate norms and residuals to verify the results
//
proc verifyResults(Ab, MatrixSpace, x) {
  initAB(Ab);

  const axmbNorm = norm(gaxpyMinus(Ab[.., 1..n], x, Ab[.., n+1..n+1]), normType.normInf);

  const a1norm   = norm(Ab[.., 1..n], normType.norm1),
        aInfNorm = norm(Ab[.., 1..n], normType.normInf),
        x1Norm   = norm(Ab[.., n+1..n+1], normType.norm1),
        xInfNorm = norm(Ab[.., n+1..n+1], normType.normInf);

  const resid1 = axmbNorm / (epsilon * a1norm * n),
        resid2 = axmbNorm / (epsilon * a1norm * x1Norm),
        resid3 = axmbNorm / (epsilon * aInfNorm * xInfNorm);

  if (printStats) {
    writeln("resid1: ", resid1);
    writeln("resid2: ", resid2);
    writeln("resid3: ", resid3);
  }

  return max(resid1, resid2, resid3) < 16.0;
}

//
// print success/failure, the execution time and the Gflop/s value
//
proc printResults(successful, execTime) {
  writeln("Validation: ", if successful then "SUCCESS" else "FAILURE");
  if (printStats) {
    writeln("Execution time = ", execTime);
    const GflopPerSec = ((2.0/3.0) * n**3 + (3.0/2.0) * n**2) / execTime * 10e-9;
    writeln("Performance (Gflop/s) = ", GflopPerSec);
  }
}

//
// simple matrix-vector multiplication, solve equation A*x-y
//
proc gaxpyMinus(A: [],
                x: [?xD],
                y: [?yD]) {
  var res: [1..n] elemType;

  forall i in 1..n do
    res[i] = (+ reduce [j in xD] (A[i,j] * x[j])) - y[n+1, i];

  return res;
}


proc dummy_calls() {  
  	
  writeln("dummy_calls() starts...");
  
  var n = 8;
  var blkSize = 2;
  	
  const matVectSpace: domain(2, indexType) dmapped 
		  BlockCyclic(startIdx=(1, 1), (blkSize, blkSize)) = [1..n, 1..n+1];	
  var ab : [matVectSpace] elemType;  // the matrix A and vector b
  var piv: [1..n] indexType;         // a vector of pivot values
  
  var abWrapper = new ArrayWrapper(elemType, 2, ab);
  impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
		  abWrapper, 
		  impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(
				  abWrapper, 2, 2) + 125.0,
		  2, 2);
  for (i, j) in matVectSpace do {
	var newVal = 0;
	if (i < j) {
	  newVal = 10 * i + j;	
	} else {
	  newVal = i + 10 * j;
	}
	impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(
	  		  abWrapper, newVal, i, j);  
  }

  var pivWrapper = new ArrayWrapper(indexType, 1, piv);
  for i in [1..n] do {
    impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(
		  pivWrapper, 
		  impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(
  				pivWrapper, i) + i,
  		  i);
  }
  
  writeln("dummy_calls() ends.");
}

