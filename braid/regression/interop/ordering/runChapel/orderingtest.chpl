use Ordering;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

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
  else
    r = ResultType.FAIL;
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

proc test_makeColumnIMatrix() {
  magicNumber = clearstack(magicNumber);	
  tracker.writeComment("Start: makeColumnIMatrix", sidl_ex);

  var A = Ordering.IntOrderTest_static.makeColumnIMatrix(arraySize, true, sidl_ex);
  
  init_part(); run_part("A is not nil", A != nil);
  init_part(); run_part("A isColumnOrder", A.isColumnOrder());
  init_part(); run_part("A isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
  init_part(); run_part("A isColumnIMatrixTwo", Ordering.IntOrderTest_static.isColumnIMatrixTwo(A, sidl_ex));
  init_part(); run_part("A isRowIMatrixTwo", Ordering.IntOrderTest_static.isRowIMatrixTwo(A, sidl_ex));
   
  Ordering.IntOrderTest_static.ensureRow(A, sidl_ex);
  init_part(); run_part("A 1.ensureRow.isRowOrder", A.isRowOrder());
  init_part(); run_part("A 1.ensureRow.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
  
  Ordering.IntOrderTest_static.ensureColumn(A, sidl_ex);
  init_part(); run_part("A 2.ensureColumn.isColumnOrder", A.isColumnOrder());
  init_part(); run_part("A 2.ensureColumn.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  Ordering.IntOrderTest_static.ensureRow(A, sidl_ex);
  init_part(); run_part("A 3.ensureRow.isRowOrder", A.isRowOrder());
  init_part(); run_part("A 3.ensureRow.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  Ordering.IntOrderTest_static.ensureColumn(A, sidl_ex);
  init_part(); run_part("A 4.ensureColumn.isColumnOrder", A.isColumnOrder());
  init_part(); run_part("A 4.ensureColumn.isIMatrixTwo", Ordering.IntOrderTest_static.isIMatrixTwo(A, sidl_ex));
    
  //  A.deleteRef();
  
  tracker.writeComment("End: makeColumnIMatrix", sidl_ex);
  magicNumber = clearstack(magicNumber);
}
test_makeColumnIMatrix();


tracker.close(sidl_ex);
