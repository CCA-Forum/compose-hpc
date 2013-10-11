/**
 * \internal
 * File:           testExpressionRoutines.c
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2013 October 11
 * \endinternal
 *
 * @file
 * @brief
 * Quick-and-dirty test suite for ExpressionRoutines.
 *
 * @htmlinclude copyright.html
 */

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "contracts.h"
#include "ExpressionRoutines.h"


/** Number of test cases. */
long g_numTests = 0;

/** Number of test cases. */
long g_numOkay = 0;


/**
 * Checks pce_in_range for a specific value.
 *
 * @param[in] var       The variable whose value is being checked.
 * @param[in] minvalue  The lowest value @a var can take on in the range.
 * @param[in] maxvalue  The highest value @a var can take on in the range.
 * @param[in] result    The expected result from the test.
 */ 
void
checkInRangeI64(
  /* in */ int64_t        var,
  /* in */ int64_t        minvalue,
  /* in */ int64_t        maxvalue,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_inrange(%ld, %ld, %ld) == %d:  ",
    g_numTests, var, minvalue, maxvalue, result);

  if (pce_in_range(var, minvalue, maxvalue) == result) 
  {
    g_numOkay +=1;
    printf("PASSED");
  } 
  else
  {
    printf("FAILED");
  } 
  printf("\n");

  return;
} /* checkInRangeLD */


/**
 * Checks pce_near_equal for a specific pair of values.
 *
 * @param[in] var     The variable whose value is being checked.
 * @param[in] value   The target equivalent value.
 * @param[in] tol     The allowable tolerance for the value range.
 * @param[in] result  The expected result from the test.
 */ 
void
checkNearEqualLD(
  /* in */ long double    var,
  /* in */ long double    value,
  /* in */ long double    tol,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. checkNearEqualLD(%Le, %Le, %Le) == %d:  ",
    g_numTests, var, value, tol, result);

  if (pce_near_equal(var, value, tol) == result) 
  {
    g_numOkay +=1;
    printf("PASSED");
  } 
  else
  {
    printf("FAILED");
  } 
  printf("\n");

  return;
} /* checkNearEqualLD */



/**
 * Test driver.  
 */
int
main(int argc, char **argv)
{
  printf("\nRunning testExpressionRoutines tests...\n");


  /* 1-4 */
  checkInRangeI64(LLONG_MIN, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MIN+1L, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MAX-1L, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MAX, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);

  printf("\n");

  /* 5-6 */
  long la = LONG_MIN;
  long lb = LONG_MAX;
  checkInRangeI64(la, LONG_MIN+1L, LONG_MAX, CONTRACTS_FALSE);
  checkInRangeI64(lb, LONG_MIN, LONG_MAX-1L, CONTRACTS_FALSE);

  printf("\n");

  /* 7-8 */
  int ia = INT_MIN;
  int ib = INT_MAX;
  checkInRangeI64(ia, INT_MIN+1, INT_MAX, CONTRACTS_FALSE);
  checkInRangeI64(ib, INT_MIN, INT_MAX-1, CONTRACTS_FALSE);

  printf("\n");

  /* 9-14 */
  long double ldtol = 0.0000001L;
  checkNearEqualLD(LDBL_MIN, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MIN+ldtol, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MIN+(2.0L*ldtol), LDBL_MIN, ldtol, CONTRACTS_FALSE);
  checkNearEqualLD(LDBL_MAX/(1.0L+ldtol), LDBL_MAX, ldtol, CONTRACTS_FALSE);
  checkNearEqualLD(LDBL_MAX-ldtol, LDBL_MAX, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MAX, LDBL_MAX, ldtol, CONTRACTS_TRUE);

  printf("\n");

  /* 15-18 */
  float ftol = 0.0000001f;
  float fa = FLT_MIN;
  float fb = FLT_MAX;
  checkNearEqualLD(fa, FLT_MIN+ftol, ftol, CONTRACTS_TRUE);
  checkNearEqualLD(fa, FLT_MIN+(2.0f*ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualLD(fb, FLT_MAX/(1.0f+ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualLD(fb, FLT_MAX-ftol, ftol, CONTRACTS_TRUE);

  printf("\n");

  /* 19-22 */
  double dtol = 0.0000001;
  double da = DBL_MIN;
  double db = DBL_MAX;
  checkNearEqualLD(da, DBL_MIN+dtol, dtol, CONTRACTS_TRUE);
  checkNearEqualLD(da, DBL_MIN+(2.0*dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualLD(db, DBL_MAX/(1.0+dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualLD(db, DBL_MAX-dtol, dtol, CONTRACTS_TRUE);


  printf("\n..%ld passed out of %ld cases\n", g_numOkay, g_numTests);
  printf("\n\nTEST SUITE %s\n", (g_numOkay==g_numTests) ? "PASSED" : "FAILED");

  return 0;
} /* main */ 
