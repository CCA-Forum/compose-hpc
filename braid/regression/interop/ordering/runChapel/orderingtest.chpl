use Ordering;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

var failed: bool = false;
var part_no: int = 0;
var sidl_ex: BaseException = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);

proc init_part() {
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part " + part_no, sidl_ex);
}

proc run_part(result: bool) {
  run_part("", result);
}

proc run_part(msg: string, result: bool) {
  if (msg.length > 0) {
    tracker.writeComment(msg, sidl_ex);
  }
  var r: ResultType;
  if (result) then
    r = ResultType.PASS;
  else {
    r = ResultType.FAIL;
    failed = true;
  }
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("End of part " + part_no, sidl_ex);
}

/**
 * Fill the stack with random junk.
 */
proc clearstack(magicNumber: int): int {
  return magicNumber;
}

const TEST_SIZE: int(32) = 345; /* size of one dimensional arrays */
const TEST_DIM1: int(32) =  17; /* first dimension of 2-d arrays */
const TEST_DIM2: int(32) =  13; /* second dimension of 2-d arrays */

const arraySize: int(32) =   7;

var magicNumber = 13;
tracker.setExpectations(-1, sidl_ex);
// tracker.setExpectations(32);

proc test_makeColumnIMatrix_1() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_makeColumnIMatrix_1", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = 
		  Ordering.IntOrderTest_static.makeColumnIMatrix(arraySize, true, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
  init_part(); run_part(" isColumnIMatrixTwo", Ordering.IntOrderTest_static.isColumnIMatrixTwo(A, sidl_ex));
  init_part(); run_part(" isRowIMatrixTwo", Ordering.IntOrderTest_static.isRowIMatrixTwo(A, sidl_ex));
   
  Ordering.IntOrderTest_static.ensureRow(A, sidl_ex);
  init_part(); run_part(" 1.ensureRow.isRowOrder", A.isRowOrder());
  init_part(); run_part(" 1.ensureRow.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
  
  Ordering.IntOrderTest_static.ensureColumn(A, sidl_ex);
  init_part(); run_part(" 2.ensureColumn.isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" 2.ensureColumn.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  Ordering.IntOrderTest_static.ensureRow(A, sidl_ex);
  init_part(); run_part(" 3.ensureRow.isRowOrder", A.isRowOrder());
  init_part(); run_part(" 3.ensureRow.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  Ordering.IntOrderTest_static.ensureColumn(A, sidl_ex);
  init_part(); run_part(" 4.ensureColumn.isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" 4.ensureColumn.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_makeColumnIMatrix_1", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeColumnIMatrix_1();


proc test_makeColumnIMatrix_2() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_makeColumnIMatrix_2", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = 
		  Ordering.IntOrderTest_static.makeColumnIMatrix(arraySize, false, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_makeColumnIMatrix_2", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeColumnIMatrix_2();


proc test_makeRowIMatrix_1() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_makeRowIMatrix_1", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = 
		  Ordering.IntOrderTest_static.makeRowIMatrix(arraySize, true, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isRowOrder", A.isRowOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
  init_part(); run_part(" isColumnIMatrixTwo", Ordering.IntOrderTest_static.isColumnIMatrixTwo(A, sidl_ex));
  init_part(); run_part(" isRowIMatrixTwo", Ordering.IntOrderTest_static.isRowIMatrixTwo(A, sidl_ex));
   
  //  A.deleteRef();
  
  tracker.writeComment("End: test_makeRowIMatrix_1", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeRowIMatrix_1();

proc test_makeRowIMatrix_2() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_makeRowIMatrix_2", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = 
		  Ordering.IntOrderTest_static.makeRowIMatrix(arraySize, false, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isRowOrder", A.isRowOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_makeRowIMatrix_2", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeRowIMatrix_2();


proc test_createColumnIMatrix_1() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_createColumnIMatrix_1", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = nil;
  Ordering.IntOrderTest_static.createColumnIMatrix(arraySize, true, A, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_createColumnIMatrix_1", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_createColumnIMatrix_1();

proc test_createColumnIMatrix_2() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_createColumnIMatrix_2", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = nil;
  Ordering.IntOrderTest_static.createColumnIMatrix(arraySize, false, A, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isColumnOrder", A.isColumnOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_createColumnIMatrix_2", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_createColumnIMatrix_2();


proc test_createRowIMatrix_1() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_createRowIMatrix_1", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = nil;
  Ordering.IntOrderTest_static.createRowIMatrix(arraySize, true, A, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isRowOrder", A.isRowOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_createRowIMatrix_1", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_createRowIMatrix_1();

proc test_createRowIMatrix_2() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_createRowIMatrix_2", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = nil;
  Ordering.IntOrderTest_static.createRowIMatrix(arraySize, false, A, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isRowOrder", A.isRowOrder());
  init_part(); run_part(" isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_createRowIMatrix_2", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_createRowIMatrix_2();


proc test_makeIMatrix() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_makeIMatrix", sidl_ex);

  var A: sidl.Array(int(32), sidl_int__array) = 
		  Ordering.IntOrderTest_static.makeIMatrix(arraySize, true, sidl_ex);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isColumnIMatrixFour", Ordering.IntOrderTest_static.isColumnIMatrixFour(A, sidl_ex));
  init_part(); run_part(" isRowIMatrixFour", Ordering.IntOrderTest_static.isRowIMatrixFour(A, sidl_ex));
  init_part(); run_part(" isIMatrixFour", Ordering.IntOrderTest_static.isIMatrixFour(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_makeIMatrix", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeIMatrix();

proc test_chapelIMatrix() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_chapelIMatrix", sidl_ex);

  // make1DIMatrix()
  var chplArray: [1 .. #arraySize] int(32);
  var chplArray_rank = chplArray.domain.rank;
  var chplArray_lus = computeLowerUpperAndStride(chplArray);
  var chplArray_lower = chplArray_lus(0);
  var chplArray_upper = chplArray_lus(1);
  var chplArray_stride = chplArray_lus(2);
              
  var wrappedArray: sidl_int__array = sidl_int__array_borrow(
        int_ptr(chplArray(chplArray.domain.low)),
        chplArray_rank,
        chplArray_lower[1],
        chplArray_upper[1],
        chplArray_stride[1]);
  var A: sidl.Array(int(32), sidl_int__array) = 
		  new Array(chplArray.eltType, sidl_int__array, wrappedArray);
  
  init_part(); run_part(" is not nil", A != nil);
  init_part(); run_part(" isIMatrixOne", Ordering.IntOrderTest_static.isIMatrixOne(A, sidl_ex));
  init_part(); run_part(" isColumnIMatrixOne", Ordering.IntOrderTest_static.isColumnIMatrixOne(A, sidl_ex));
  init_part(); run_part(" isRowIMatrixOne", Ordering.IntOrderTest_static.isRowIMatrixOne(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: test_chapelIMatrix", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_chapelIMatrix();

proc test_isSliceWorking() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: test_isSliceWorking", sidl_ex);
  init_part(); run_part(" isSliceWorking(true)", Ordering.IntOrderTest_static.isSliceWorking(true, sidl_ex));
  init_part(); run_part(" isSliceWorking(false)", Ordering.IntOrderTest_static.isSliceWorking(false, sidl_ex));
  tracker.writeComment("End: test_isSliceWorking", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_isSliceWorking();

tracker.close(sidl_ex);

if (failed) then
  exit(1);
