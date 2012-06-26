use vect;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

// FIXME
var ALLOW_MEMORY_LEAK_CHECKS = true;
var statsFile = "VUtils.stats";
var failed: bool = false;
var part_no: int(32) = 0;
var sidl_ex: BaseInterface = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);

proc init_part(msg: string)
{
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part "+part_no, sidl_ex);

  if (msg.length > 0) {
    tracker.writeComment(msg, sidl_ex);
  }
}

proc run_part(result: bool)
{
  run_part("", result);
}

proc run_part(result, exp_result, sidl_ex, expectation)
{
  var exp_type: string;
  select expectation {
    when vect.ExpectExcept.PreExp  do exp_type = 'sidl.PreViolation';
    when vect.ExpectExcept.PostExp do exp_type = 'sidl.PostViolation';
    when vect.ExpectExcept.DBZExp  do exp_type = 'vect.vDivByZeroExcept';
    when vect.ExpectExcept.NVEExp  do exp_type = 'vect.vNegValExcept';
    when vect.ExpectExcept.ExcExp  do exp_type = 'vect.vExcept';
    otherwise exp_type = '';
  }
  var ex: sidl.BaseInterface;

  var r: ResultType;
  if (result == exp_result) then {
    if expectation == vect.ExpectExcept.NoneExp then
      if sidl_ex == nil then
	r = ResultType.PASS;
      else {
	r = ResultType.FAIL;
	failed = true;
      }
    else if sidl_ex.isType(exp_type, ex) then
      r = ResultType.PASS;
    else {
      r = ResultType.FAIL;
      failed = true;
    }
  } else {
    r = ResultType.FAIL;
    failed = true;
  }
  tracker.endPart(part_no, r, ex);
  tracker.writeComment("End of part "+part_no, ex);
}

proc run_parta(result:opaque, tol, exp_result:opaque, expect_equal:bool, 
	       sidl_ex, expectation)
{
  var exp_type: string;
  select expectation {
    when vect.ExpectExcept.PreExp  do exp_type = 'sidl.PreViolation';
    when vect.ExpectExcept.PostExp do exp_type = 'sidl.PostViolation';
    when vect.ExpectExcept.DBZExp  do exp_type = 'vect.vDivByZeroExcept';
    when vect.ExpectExcept.NVEExp  do exp_type = 'vect.vNegValExcept';
    when vect.ExpectExcept.ExcExp  do exp_type = 'vect.vExcept';
    otherwise exp_type = '';
  }
  var ex: sidl.BaseInterface;

  var r: ResultType;
  if expectation == vect.ExpectExcept.NoneExp then {
    if (sidl_ex == nil && 
	((IS_NULL(result) && IS_NULL(exp_result)) ||
	 (expect_equal  &&  vect.Utils_static.vuAreEqual(result, exp_result, tol, ex)) ||
	 (!expect_equal && !vect.Utils_static.vuAreEqual(result, exp_result, tol, ex)))) then {
      r = ResultType.PASS;
    } else {
      r = ResultType.FAIL;
      failed = true;
    } 
  } else {
    if sidl_ex.isType(exp_type, ex) then {
      r = ResultType.PASS;
    } else {
      r = ResultType.FAIL;
      failed = true;
    }
  }
  tracker.endPart(part_no, r, ex);
  tracker.writeComment("End of part "+part_no, ex);
}

proc run_part(result:real(64), tol, exp_result, sidl_ex, expectation)
{
  var exp_type: string;
  select expectation {
    when vect.ExpectExcept.PreExp  do exp_type = 'sidl.PreViolation';
    when vect.ExpectExcept.PostExp do exp_type = 'sidl.PostViolation';
    when vect.ExpectExcept.DBZExp  do exp_type = 'vect.vDivByZeroExcept';
    when vect.ExpectExcept.NVEExp  do exp_type = 'vect.vNegValExcept';
    when vect.ExpectExcept.ExcExp  do exp_type = 'vect.vExcept';
    otherwise exp_type = '';
  }
  var ex: sidl.BaseInterface;

  var r: ResultType;
  if expectation == vect.ExpectExcept.NoneExp then {
    if (abs(result - exp_result) <= abs(tol)) then {
      if sidl_ex == nil then
	r = ResultType.PASS;
       else {
	r = ResultType.FAIL;
	failed = true;
       }
    } else {
      r = ResultType.FAIL;
      failed = true;
    }
  } else {
    if sidl_ex.isType(exp_type, ex) then
      r = ResultType.PASS;
    else {
      r = ResultType.FAIL;
      failed = true;
    }
  }
  tracker.endPart(part_no, r, ex);
  tracker.writeComment("End of part "+part_no, ex);
}


// START: Assert methods
param EPSILON = 0.0001;

var i, j: int(32);
var max_size:int(32) = 6;
var sqrt_size = sqrt(1.0*max_size);
var tol       = 1.0e-9;
var ntol      = -1.0e-9;
var val       = 1.0/sqrt_size;
var nval      = -1.0/sqrt_size;

var t	= sidl.double_array.create2dCol(max_size, max_size);
var u	= sidl.double_array.create1d(max_size);
var u1  = sidl.double_array.create1d(max_size+1);
var nu  = sidl.double_array.create1d(max_size);
var z 	= sidl.double_array.create1d(max_size);
var null_array: opaque;
for i in 0..#max_size do {
  u(2)[i:int(32)] = val;
  u1(2)[i:int(32)] = val;
  nu(2)[i:int(32)] = nval;
  z(2)[i:int(32)] = 0;

  for j in 0..#max_size do {
    t(2)[i:int(32),j:int(32)] = val; 
  }
 }

u1(2)[max_size] = val;


if ALLOW_MEMORY_LEAK_CHECKS then
  /*
   * The actual number of tests differs depending on whether the calls
   * that pass the regression tests but leak memory should be enabled
   * or not.  The ten tests leaking memory occur whenever the implementation
   * attempts to return a different sized output array (or matrix).  
   *
   * The solution _may_ be to change the corresponding check routines
   * to free the memory (on the postcondition violation) and return a
   * null pointer.  However, that will not protect C++ clients from
   * leaking memory from an errant implementation that does not have
   * such a contract check.
   */
   tracker.setExpectations(128, sidl_ex);
 else
   tracker.setExpectations(118, sidl_ex);
/* ALLOW_MEMORY_LEAK_CHECKS */

tracker.writeComment("*** ENABLE FULL CONTRACT CHECKING ***", sidl_ex);
sidl.EnfPolicy_static.setEnforceAll(sidl.ContractClass.ALLCLASSES, true, sidl_ex);

/* vuIsZero() set */
init_part("ensuring the zero vector is the zero vector");
run_part(vect.Utils_static.vuIsZero(z(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the unit vector is not the zero vector");
run_part(vect.Utils_static.vuIsZero(u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuIsZero() a null array");
run_part(vect.Utils_static.vuIsZero(null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuIsZero() a 2D array");
run_part(vect.Utils_static.vuIsZero(t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuIsZero() a negative tolerance");
run_part(vect.Utils_static.vuIsZero(z(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuIsUnit() set */
init_part("ensuring the unit vector is the unit vector");
run_part(vect.Utils_static.vuIsUnit(u(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the zero vector is not the unit vector");
run_part(vect.Utils_static.vuIsUnit(z(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuIsUnit() a null array");
run_part(vect.Utils_static.vuIsUnit(null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuIsUnit() a 2D array");
run_part(vect.Utils_static.vuIsUnit(t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuIsUnit() a negative tolerance");
run_part(vect.Utils_static.vuIsUnit(u(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuAreEqual() set */
init_part("ensuring the unit and zero vectors are not equal");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, z(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the unit vector is equal to itself");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, u(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuAreEqual() a null 1st array");
run_part(vect.Utils_static.vuAreEqual(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreEqual() a 2D 1st array");
run_part(vect.Utils_static.vuAreEqual(t(1).generic, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreEqual() a null 2nd array");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreEqual() a 2D 2nd array");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreEqual() different sized arrays");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, u1(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreEqual() a negative tolerance");
run_part(vect.Utils_static.vuAreEqual(u(1).generic, u(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuAreOrth() set */
init_part("ensuring the unit and zero vectors are orthogonal");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, z(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the unit and negative unit vectors are not orthogonal");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, nu(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuAreOrth() a null 1st array");
run_part(vect.Utils_static.vuAreOrth(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() a 2D 1st array");
run_part(vect.Utils_static.vuAreOrth(t(1).generic, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() a null 2nd array");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() a 2D 2nd array");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() different sized unit arrays");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, u1(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() a negative tolerance");
run_part(vect.Utils_static.vuAreOrth(u(1).generic, u(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuAreOrth() 2D arrays in both arguments");
run_part(vect.Utils_static.vuAreOrth(t(1).generic, t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuSchwarzHolds() set */
init_part("ensuring schwarz holds for the unit and zero vectors");
run_part(vect.Utils_static.vuSchwarzHolds(u(1).generic, z(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuSchwarzHolds() a null 1st array");
run_part(vect.Utils_static.vuSchwarzHolds(null_array, z(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSchwarzHolds() a 2D 1st array");
run_part(vect.Utils_static.vuSchwarzHolds(t(1).generic, z(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSchwarzHolds() a null 2nd array");
run_part(vect.Utils_static.vuSchwarzHolds(z(1).generic, null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSchwarzHolds() a 2D 2nd array");
run_part(vect.Utils_static.vuSchwarzHolds(u(1).generic, t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSchwarzHolds() different sized unit arrays");
run_part(vect.Utils_static.vuSchwarzHolds(u(1).generic, u1(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSchwarzHolds() a negative tolerance");
run_part(vect.Utils_static.vuSchwarzHolds(u(1).generic, z(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuTriIneqHolds() set */
init_part("ensuring the triangle inequality holds for the unit and zero vectors");
run_part(vect.Utils_static.vuTriIneqHolds(u(1).generic, z(1).generic, tol, sidl_ex),
         true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuTriIneqHolds() a null 1st array");
run_part(vect.Utils_static.vuTriIneqHolds(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuTriIneqHolds() a 2D 1st array");
run_part(vect.Utils_static.vuTriIneqHolds(t(1).generic, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuTriIneqHolds() a null 2nd array");
run_part(vect.Utils_static.vuTriIneqHolds(u(1).generic, null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuTriIneqHolds() a 2D 2nd array");
run_part(vect.Utils_static.vuTriIneqHolds(u(1).generic, t(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuTriIneqHolds() different sized unit vectors");
run_part(vect.Utils_static.vuTriIneqHolds(u(1).generic, u1(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuTriIneqHolds() a negative tolerance");
run_part(vect.Utils_static.vuTriIneqHolds(u(1).generic, u(1).generic, ntol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.PreExp);

/* vuNorm() set */
init_part("ensuring the unit vector norm is 1.0");
run_part(vect.Utils_static.vuNorm(u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         1.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNorm() a null vector");
run_part(vect.Utils_static.vuNorm(null_array, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNorm() a 2D array");
run_part(vect.Utils_static.vuNorm(t(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNorm() a negative tolerance");
run_part(vect.Utils_static.vuNorm(u(1).generic, ntol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNorm() badness level for negative result");
run_part(vect.Utils_static.vuNorm(u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         -5.0, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuNorm() badness level for positive result with zero vector");
run_part(vect.Utils_static.vuNorm(z(1).generic, tol, vect.BadLevel.PosRes, sidl_ex),
         5.0, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuNorm() badness level for zero result with non-zero vector");
run_part(vect.Utils_static.vuNorm(u(1).generic, tol, vect.BadLevel.ZeroRes, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PostExp);

/* vuDot() set */
init_part("ensuring the dot of the unit and zero vectors is 0.0");
run_part(vect.Utils_static.vuDot(u(1).generic, z(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() a null 1st array");
run_part(vect.Utils_static.vuDot(null_array, u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() a 2D 1st array");
run_part(vect.Utils_static.vuDot(t(1).generic, u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() a null 2nd array");
run_part(vect.Utils_static.vuDot(u(1).generic, null_array, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() a 2D 2nd array");
run_part(vect.Utils_static.vuDot(u(1).generic, t(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() different sized unit vectors");
run_part(vect.Utils_static.vuDot(u(1).generic, u1(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() a negative tolerance");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, ntol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() a 2D arrays in both arguments");
run_part(vect.Utils_static.vuDot(t(1).generic, t(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() badness level for negative result with u=v");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         -5.0, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuDot() badness level for positive result with u=v=0");
run_part(vect.Utils_static.vuDot(z(1).generic, z(1).generic, tol, vect.BadLevel.PosRes, sidl_ex),
         5.0, sidl_ex, vect.ExpectExcept.PostExp);

/* /\* vuProduct() set *\/ */
init_part("ensuring the product of 1 and the unit vector is the unit vector");
run_parta(vect.Utils_static.vuProduct(1.0, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the product of 2 and the unit vector is not the unit vector");
run_parta(vect.Utils_static.vuProduct(2.0, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuProduct() a null array");
run_parta(vect.Utils_static.vuProduct(0.0, null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuProduct() a 2D array");
run_parta(vect.Utils_static.vuProduct(1.0, t(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuProduct() badness level for null result");
run_parta(vect.Utils_static.vuProduct(1.0, u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
if ALLOW_MEMORY_LEAK_CHECKS then {
  init_part("passing vuProduct() badness level for 2D result)");
run_parta(vect.Utils_static.vuProduct(1.0, u(1).generic, vect.BadLevel.TwoDRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
  init_part("passing vuProduct() badness level for wrong size result");
run_parta(vect.Utils_static.vuProduct(1.0, u(1).generic, vect.BadLevel.WrongSizeRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
} /* ALLOW_MEMORY_LEAK_CHECKS */

/* vuNegate() set */
init_part("ensuring the negation of the unit vector is its negative");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the negation of the unit vector is not the unit vector");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNegate() a null array");
run_parta(vect.Utils_static.vuNegate(null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNegate() a 2D array");
run_parta(vect.Utils_static.vuNegate(t(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNegate() badness level for null result");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
if ALLOW_MEMORY_LEAK_CHECKS then {
init_part("passing vuNegate() badness level for 2D result)");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.TwoDRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuNegate() badness level for wrong size result");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.WrongSizeRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
} /* ALLOW_MEMORY_LEAK_CHECKS */

/* vuNormalize() set */
init_part("ensuring normalize of the unit vector is itself");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring normalize of the unit vector is not its negative");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring normalize of the zero vector raises a DBZ exception");
run_parta(vect.Utils_static.vuNormalize(z(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, z(1).generic, true, sidl_ex, vect.ExpectExcept.DBZExp);
init_part("passing vuNormalize() a null array");
run_parta(vect.Utils_static.vuNormalize(null_array, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNormalize() a 2D array");
run_parta(vect.Utils_static.vuNormalize(t(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNormalize() a negative tolerance using the unit vector");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, ntol, vect.BadLevel.NoVio, sidl_ex),
         ntol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuNormalize() a badness level for null result");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
if ALLOW_MEMORY_LEAK_CHECKS then {
init_part("passing vuNormalize() a badness level for 2D result");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.TwoDRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuNormalize() a badness level for wrong size result");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.WrongSizeRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
} /* ALLOW_MEMORY_LEAK_CHECKS */

/* vuSum() set (NOTE: tolerance not relevant to vuSum() API.) */
init_part("ensuring the sum of the unit and zero vectors is the unit vector");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the sum of unit and zero vectors is not the negative of the unit");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuSum() a null 1st array");
run_parta(vect.Utils_static.vuSum(null_array, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSum() a 2D 1st array");
run_parta(vect.Utils_static.vuSum(t(1).generic, null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSum() a null 2nd array");
run_parta(vect.Utils_static.vuSum(u(1).generic, null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSum() a 2D as second");
run_parta(vect.Utils_static.vuSum(u(1).generic, t(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSum() different sized unit vectors");
run_parta(vect.Utils_static.vuSum(u(1).generic, u1(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, z(1).generic, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuSum() badness level for null result");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
if ALLOW_MEMORY_LEAK_CHECKS then {
init_part("passing vuSum() badness level for 2D result");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.TwoDRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuSum() badness level for wrong size result");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.WrongSizeRes, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
} /* ALLOW_MEMORY_LEAK_CHECKS */

/* vuDiff() set (NOTE: tolerance not relevant to vuDiff() API.) */
init_part("ensuring the diff of the zero and unit vectors is the negative unit vector");
run_parta(vect.Utils_static.vuDiff(z(1).generic, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the diff of the unit and zero vectors is the unit vector");
run_parta(vect.Utils_static.vuDiff(u(1).generic, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("ensuring the diff of the unit and zero vectors is not the neg unit vector");
run_parta(vect.Utils_static.vuDiff(u(1).generic, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDiff() a null 1st array");
run_parta(vect.Utils_static.vuDiff(null_array, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDiff() a 2D 1st array");
run_parta(vect.Utils_static.vuDiff(t(1).generic, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDiff() a null 2nd array");
run_parta(vect.Utils_static.vuDiff(u(1).generic, null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDiff() a 2D 2nd array");
run_parta(vect.Utils_static.vuDiff(u(1).generic, t(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDiff() different sized vectors");
run_parta(vect.Utils_static.vuDiff(u(1).generic, u1(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, u(1).generic, true, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDiff() badness level for null result");
run_parta(vect.Utils_static.vuDiff(z(1).generic, u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
if ALLOW_MEMORY_LEAK_CHECKS then {
init_part("passing vuDiff() badness level for 2D result");
run_parta(vect.Utils_static.vuDiff(z(1).generic, u(1).generic, vect.BadLevel.TwoDRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
init_part("passing vuDiff() badness level for wrong size result");
run_parta(vect.Utils_static.vuDiff(z(1).generic, u(1).generic, vect.BadLevel.WrongSizeRes, sidl_ex),
         tol, nu(1).generic, true, sidl_ex, vect.ExpectExcept.PostExp);
} /* ALLOW_MEMORY_LEAK_CHECKS */

//vect.Utils._dump_stats_static(statsFile, "After full checking");

/****************************************************************
 * Now check preconditions only.  Only need three checks:
 *   1) successful execution;
 *   2) precondition violation that is not caught but is
 *      okay anyway; and
 *   3) postcondition violation that is caught.
 ****************************************************************/
tracker.writeComment("*** ENABLE PRECONDITION ENFORCEMENT ONLY ***", sidl_ex);
sidl.EnfPolicy_static.setEnforceAll(sidl.ContractClass.PRECONDS, false, sidl_ex);

init_part("ensuring the dot product of the unit and zero vectors is 0.0");
run_part(vect.Utils_static.vuDot(u(1).generic, z(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, 0.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() a negative tolerance");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, ntol, vect.BadLevel.NoVio, sidl_ex),
         ntol, 1.0, sidl_ex, vect.ExpectExcept.PreExp);
init_part("passing vuDot() badness level for negative result with u=v");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         tol, -5.0, sidl_ex, vect.ExpectExcept.NoneExp);

//vect.Utils._dump_stats_static(statsFile, "After precondition checking");

/****************************************************************
 * Now check postconditions only.  Only need three checks:
 *   1) successful execution;
 *   2) precondition violation that gets caught; and
 *   3) postcondition violation that is not caught.
 ****************************************************************/
tracker.writeComment("*** ENABLE POSTCONDITION ENFORCEMENT ONLY ***", sidl_ex);
sidl.EnfPolicy_static.setEnforceAll(sidl.ContractClass.POSTCONDS, false, sidl_ex);

init_part("ensuring the dot product of the unit and zero vectors is 0.0");
run_part(vect.Utils_static.vuDot(u(1).generic, z(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, 0.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() a negative tolerance");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, ntol, vect.BadLevel.NoVio, sidl_ex),
         ntol, 1.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() badness level for negative result with u=v");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         tol, -5.0, sidl_ex, vect.ExpectExcept.PostExp);

//vect.Utils._dump_stats_static(statsFile, "After Postcondition checking");

/****************************************************************
 * Now make sure contract violations are not caught when contract
 * enforcement turned off.  Do this for each type of violation
 * for every method.
 ****************************************************************/
tracker.writeComment("*** DISABLE ALL CONTRACT ENFORCEMENT ***", sidl_ex);
sidl.EnfPolicy_static.setEnforceNone(false, sidl_ex);

init_part("passing vuIsZero() a null array - no precondition violation");
run_part(vect.Utils_static.vuIsZero(null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuIsUnit() a null array - no precondition violation");
run_part(vect.Utils_static.vuIsUnit(null_array, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuAreEqual() a null 1st array - no precondition violation");
run_part(vect.Utils_static.vuAreEqual(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuAreOrth() a null 1st array - no precondition violation");
run_part(vect.Utils_static.vuAreOrth(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuSchwarzHolds() a null 1st array - no precondition violation");
run_part(vect.Utils_static.vuSchwarzHolds(null_array, z(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing() vuTriIneqHolds() a null 1st array - no precondition violation");
run_part(vect.Utils_static.vuTriIneqHolds(null_array, u(1).generic, tol, sidl_ex),
         false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing() vuNorm() a null vector - no precondition violation");
run_part(vect.Utils_static.vuNorm(null_array, tol, vect.BadLevel.NoVio, sidl_ex),
         0.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNorm() badness level for negative result - no post violation");
run_part(vect.Utils_static.vuNorm(u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         tol, -5.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() a null 1st array - no precondition violation");
run_part(vect.Utils_static.vuDot(null_array, u(1).generic, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, 0.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDot() badness level for negative result with u=v - no post vio.");
run_part(vect.Utils_static.vuDot(u(1).generic, u(1).generic, tol, vect.BadLevel.NegRes, sidl_ex),
         tol, -5.0, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuProduct() a null array - no precondition violation");
run_parta(vect.Utils_static.vuProduct(0.0, null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuProduct() badness level for null result - no post violation");
run_parta(vect.Utils_static.vuProduct(1.0, u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNegate() a null array - no precondition violation");
run_parta(vect.Utils_static.vuNegate(null_array, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNegate() badness level for null result - no post violation");
run_parta(vect.Utils_static.vuNegate(u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNormalize() a null array - no precondition violation");
run_parta(vect.Utils_static.vuNormalize(null_array, tol, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuNormalize() a badness level for null result - no post violation");
run_parta(vect.Utils_static.vuNormalize(u(1).generic, tol, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuSum() a null 1st array - no precondition violation");
run_parta(vect.Utils_static.vuSum(null_array, z(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuSum() a badness level for null result - no post violation");
run_parta(vect.Utils_static.vuSum(u(1).generic, z(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, u(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDiff() a null 1st array - no precondition violation");
run_parta(vect.Utils_static.vuDiff(null_array, u(1).generic, vect.BadLevel.NoVio, sidl_ex),
         tol, null_array, true, sidl_ex, vect.ExpectExcept.NoneExp);
init_part("passing vuDiff() badness level for null result - no post violation");
run_parta(vect.Utils_static.vuDiff(z(1).generic, u(1).generic, vect.BadLevel.NullRes, sidl_ex),
         tol, nu(1).generic, false, sidl_ex, vect.ExpectExcept.NoneExp);

//vect.Utils._dump_stats_static(statsFile, "After no checking");

t(1).deleteRef();
u(1).deleteRef();
u1(1).deleteRef();
nu(1).deleteRef();
z(1).deleteRef();

tracker.close(sidl_ex);
tracker.deleteRef(sidl_ex);
