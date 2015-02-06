/**
 * \internal
 * File:           testExpressionRoutines.c
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2015 February 5
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
 * Free any allocated memory in the provided array.
 */
void
cleanup(
  /* in */ void**   arr,
  /* in */ int64_t  sz)
{
  for (int64_t i = 0; i < sz; i++) {
    if (arr[i] != NULL) 
    {
        free((void*)arr[i]);
    }
  }
  free((void*)arr);
} /* cleanup */


/**
 * Checks pce_all_null for an expected result.
 *
 * @param[in] arr     The array whose (pointer) contents are being checked.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllNull(
  /* in */ void**         arr,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_null(%s, %d) == %d:  ",
    g_numTests, desc, num, result);

  if (pce_all_null(arr, num) == result) 
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
} /* checkAllNull */


/**
 * Checks pce_any_null for an expected result.
 *
 * @param[in] arr     The array whose (pointer) contents are being checked.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyNull(
  /* in */ void**         arr,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_null(%s, %d) == %d:  ",
    g_numTests, desc, num, result);

  if (pce_any_null(arr, num) == result) 
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
} /* checkAnyNull */



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
  printf("%3ld. pce_in_range(%ld, %ld, %ld) == %d:  ",
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
} /* checkInRangeI64 */


/**
 * Checks pce_range for a specific value.
 *
 * @param[in] var       The variable whose value is being checked.
 * @param[in] minvalue  The lowest value @a var can take on in the range.
 * @param[in] maxvalue  The highest value @a var can take on in the range.
 * @param[in] tol       The tolerance for the min and max values.
 * @param[in] result    The expected result from the test.
 */ 
void
checkRangeF(
  /* in */ long double    var,
  /* in */ long double    minvalue,
  /* in */ long double    maxvalue,
  /* in */ long double    tol,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_range(%Le, %Le, %Le, %Le) == %d:  ",
    g_numTests, var, minvalue, maxvalue, tol, result);

  if (pce_range(var, minvalue, maxvalue, tol) == result) 
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
} /* checkRangeF */


/**
 * Checks pce_max for a specific pair of float values.
 *
 * @param[in] a       The first value.
 * @param[in] b       The second value.
 * @param[in] expRes  The expected maximum value.
 * @param[in] result  The expected result from the test.
 */ 
void
checkMaxF(
  /* in */ long double    a,
  /* in */ long double    b,
  /* in */ long double    expRes,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. (checkMaxF(%Le, %Le) == %Le) == %d:  ", g_numTests, a, b, 
    expRes, result);

  if ((pce_max(a, b) == expRes) == result) 
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
} /* checkMaxF */


/**
 * Checks pce_max for a specific pair of integer values.
 *
 * @param[in] a       The first value.
 * @param[in] b       The second value.
 * @param[in] expRes  The expected maximum value.
 * @param[in] result  The expected result from the test.
 */ 
void
checkMaxI(
  /* in */ int64_t        a,
  /* in */ int64_t        b,
  /* in */ int64_t        expRes,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. (checkMaxI(%ld, %ld) == %ld) == %d:  ", g_numTests, a, b, 
    expRes, result);

  if ((pce_max(a, b) == expRes) == result) 
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
} /* checkMaxI */


/**
 * Checks pce_min for a specific pair of float values.
 *
 * @param[in] a       The first value.
 * @param[in] b       The second value.
 * @param[in] expRes  The expected minimum value.
 * @param[in] result  The expected result from the test.
 */ 
void
checkMinF(
  /* in */ long double    a,
  /* in */ long double    b,
  /* in */ long double    expRes,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. (checkMinF(%Le, %Le) == %Le) == %d:  ", g_numTests, a, b, 
    expRes, result);

  if ((pce_min(a, b) == expRes) == result) 
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
} /* checkMinF */


/**
 * Checks pce_min for a specific pair of integer values.
 *
 * @param[in] a       The first value.
 * @param[in] b       The second value.
 * @param[in] expRes  The expected minimum value.
 * @param[in] result  The expected result from the test.
 */ 
void
checkMinI(
  /* in */ int64_t        a,
  /* in */ int64_t        b,
  /* in */ int64_t        expRes,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. (checkMinI(%ld, %ld) == %ld) == %d:  ", g_numTests, a, b, 
    expRes, result);

  if ((pce_min(a, b) == expRes) == result) 
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
} /* checkMinI */


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
 * Executes checkAllNull cases for a character array of character arrays.
 */
void
checkPceAllNullChars()
{
  int i, sz = 20;
  const char *desc = "chcharr";
  char **chcharr = (char **)malloc((size_t)sz * sizeof(char *));

  for (i = 0; i < sz; i++) chcharr[i] = NULL;
  checkAllNull((void**)chcharr, sz, desc, CONTRACTS_TRUE);

  chcharr[sz-1] = (char *)malloc((size_t)sz * sizeof(char));
  checkAllNull((void**)chcharr, sz, desc, CONTRACTS_FALSE);

  free((void*)chcharr[sz-1]);
  chcharr[sz-1] = NULL;
  checkAllNull((void**)chcharr, sz, desc, CONTRACTS_TRUE);

  chcharr[0] = (char *)malloc((size_t)sz * sizeof(char));
  checkAllNull((void**)chcharr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)chcharr, sz);
} /* checkPceAllNullChars */


/**
 * Executes checkAllNull cases for an integer array.
 */
void
checkPceAllNullInts()
{
  int sz = 20;
  const char *desc = "intarr";
  int **intarr = (int **)malloc((size_t)sz * sizeof(int *));

  for (int i = 0; i < sz; i++) intarr[i] = NULL;
  checkAllNull((void**)intarr, sz, desc, CONTRACTS_TRUE);

  intarr[sz-1] = (int *)malloc((size_t)sz * sizeof(int));
  checkAllNull((void**)intarr, sz, desc, CONTRACTS_FALSE);

  free((void*)intarr[sz-1]);
  intarr[sz-1] = NULL;
  checkAllNull((void**)intarr, sz, desc, CONTRACTS_TRUE);

  intarr[0] = (int *)malloc((size_t)sz * sizeof(int));
  checkAllNull((void**)intarr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)intarr, sz);
} /* checkAllNullInts */


/**
 * Executes checkAllNull cases for a long double array.
 */
void
checkPceAllNullLDs()
{
  int sz = 20;
  const char *desc = "ldarr";
  long double **ldarr = (long double **)malloc((size_t)sz * 
    sizeof(long double *));

  for (int i = 0; i < sz; i++) ldarr[i] = NULL;
  checkAllNull((void**)ldarr, sz, desc, CONTRACTS_TRUE);

  ldarr[sz-1] = (long double *)malloc((size_t)sz * sizeof(long double));
  checkAllNull((void**)ldarr, sz, desc, CONTRACTS_FALSE);

  free((void*)ldarr[sz-1]);
  ldarr[sz-1] = NULL;
  checkAllNull((void**)ldarr, sz, desc, CONTRACTS_TRUE);

  ldarr[0] = (long double *)malloc((size_t)sz * sizeof(long double));
  checkAllNull((void**)ldarr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)ldarr, sz);
} /* checkAllNullLDs */


/**
 * Executes checkAnyNull cases for a character array of characters.
 */
void
checkPceAnyNullChars()
{
  int i, sz = 20;
  const char *desc = "chcharr";
  char **chcharr = (char **)malloc((size_t)sz * sizeof(char *));

  for (i = 0; i < sz; i++) chcharr[i] = NULL;
  checkAnyNull((void**)chcharr, sz, desc, CONTRACTS_TRUE);

  for (i = 1; i < sz-1; i++) {
    chcharr[i] = (char *)malloc((size_t)sz * sizeof(char));
  }
  checkAnyNull((void**)chcharr, sz, desc, CONTRACTS_TRUE);

  chcharr[0] = (char *)malloc((size_t)sz * sizeof(char));
  checkAnyNull((void**)chcharr, sz, desc, CONTRACTS_TRUE);

  chcharr[sz-1] = (char *)malloc((size_t)sz * sizeof(char));
  checkAnyNull((void**)chcharr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)chcharr, sz);
} /* checkPceAnyNullChars */


/**
 * Executes checkAnyNull cases for an integer array.
 */
void
checkPceAnyNullInts()
{
  int i, sz = 20;
  const char *desc = "intarr";
  int **intarr = (int **)malloc((size_t)sz * sizeof(int *));

  for (i = 0; i < sz; i++) intarr[i] = NULL;
  checkAnyNull((void**)intarr, sz, desc, CONTRACTS_TRUE);

  for (i = 1; i < sz-1; i++) {
      intarr[i] = (int *)malloc(sizeof(int));
  }
  checkAnyNull((void**)intarr, sz, desc, CONTRACTS_TRUE);

  intarr[0] = (int *)malloc(sizeof(int));
  checkAnyNull((void**)intarr, sz, desc, CONTRACTS_TRUE);

  intarr[sz-1] = (int *)malloc(sizeof(int));
  checkAnyNull((void**)intarr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)intarr, sz);
} /* checkAnyNullInts */


/**
 * Executes checkAnyNull cases for a long double array.
 */
void
checkPceAnyNullLDs()
{
  int i, sz = 20;
  const char *desc = "ldarr";
  long double **ldarr = (long double **)malloc((size_t)sz * 
    sizeof(long double *));

  for (i = 0; i < sz; i++) ldarr[i] = NULL;
  checkAnyNull((void**)ldarr, sz, desc, CONTRACTS_TRUE);

  for (i = 1; i < sz-1; i++) {
      ldarr[i] = (long double *)malloc(sizeof(long double));
  }
  checkAnyNull((void**)ldarr, sz, desc, CONTRACTS_TRUE);

  ldarr[0] = (long double *)malloc(sizeof(long double));
  checkAnyNull((void**)ldarr, sz, desc, CONTRACTS_TRUE);

  ldarr[sz-1] = (long double *)malloc(sizeof(long double));
  checkAnyNull((void**)ldarr, sz, desc, CONTRACTS_FALSE);

  printf("\n");

  cleanup((void**)ldarr, sz);
} /* checkAnyNullLDs */


/**
 * Executes all checkInRangeI64 cases.
 */
void
checkPceInRange()
{
  /* First use long long values */
  checkInRangeI64(LLONG_MIN, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MIN+1L, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MAX-1L, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MAX, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkInRangeI64(LLONG_MAX, LLONG_MAX, LLONG_MIN, CONTRACTS_FALSE);
  checkInRangeI64(LLONG_MIN, LLONG_MAX, LLONG_MIN, CONTRACTS_FALSE);

  printf("\n");

  /* Use long values */
  checkInRangeI64(LONG_MIN, LONG_MIN+1L, LONG_MAX, CONTRACTS_FALSE);
  checkInRangeI64(LONG_MAX, LONG_MIN, LONG_MAX-1L, CONTRACTS_FALSE);

  printf("\n");

  /* Use int values */
  checkInRangeI64(INT_MIN, INT_MIN+1, INT_MAX, CONTRACTS_FALSE);
  checkInRangeI64(INT_MAX, INT_MIN, INT_MAX-1, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceInRange */


/**
 * Executes all checkRangeF cases.
 */
void
checkPceRange()
{
  long double dtol = 0.0000001L;

  /* First use double values */
  checkRangeF(DBL_MIN, DBL_MIN, DBL_MAX, dtol, CONTRACTS_TRUE);
  checkRangeF(DBL_MIN+dtol, DBL_MIN, DBL_MAX, dtol, CONTRACTS_TRUE);
  checkRangeF(DBL_MAX-dtol, DBL_MIN, DBL_MAX, dtol, CONTRACTS_TRUE);
  checkRangeF(DBL_MIN-2.*dtol, DBL_MIN, DBL_MAX, dtol, CONTRACTS_FALSE);
  /* Note tol gets lost when added to DBL_MAX... */
  checkRangeF(DBL_MAX+2.*dtol, DBL_MIN, DBL_MAX, dtol, CONTRACTS_TRUE);
  checkRangeF(1.1*DBL_MAX, DBL_MIN, DBL_MAX, dtol, CONTRACTS_FALSE);
  checkRangeF(DBL_MAX, DBL_MAX, DBL_MIN, dtol, CONTRACTS_FALSE);
  checkRangeF(DBL_MIN, DBL_MAX, DBL_MIN, dtol, CONTRACTS_FALSE);

  printf("\n");

  /* Use double values with 0.0 tolerance */
  checkRangeF(DBL_MIN, DBL_MIN, DBL_MAX, 0.0, CONTRACTS_TRUE);
  checkRangeF(DBL_MAX, DBL_MIN, DBL_MAX, 0.0, CONTRACTS_TRUE);
  checkRangeF(DBL_MIN-dtol, DBL_MIN, DBL_MAX, 0.0, CONTRACTS_FALSE);
  checkRangeF(DBL_MAX+dtol, DBL_MIN, DBL_MAX, 0.0, CONTRACTS_TRUE);
  checkRangeF(1.1*DBL_MAX, DBL_MIN, DBL_MAX, 0.0, CONTRACTS_FALSE);

  printf("\n");

  /* Use long double values */
  checkRangeF(LDBL_MIN, LDBL_MIN+2.*dtol, LDBL_MAX, dtol, CONTRACTS_FALSE);
  /* Note tol gets lost when added to DBL_MAX... */
  checkRangeF(LDBL_MAX, LDBL_MIN, LDBL_MAX-DBL_MIN, dtol, CONTRACTS_TRUE);
  /* Note DBL_MAX gets lost when added to DBL_MAX... */
  checkRangeF(LDBL_MAX, LDBL_MIN, LDBL_MAX-DBL_MAX, dtol, CONTRACTS_TRUE);
  checkRangeF(LDBL_MAX, LDBL_MIN, 0.9*LDBL_MAX, dtol, CONTRACTS_FALSE);

  printf("\n");

  /* Use long double values with 0.0 tolerance */
  checkRangeF(LDBL_MIN, LDBL_MIN+dtol, LDBL_MAX, 0.0, CONTRACTS_FALSE);
  /* Note tol gets lost when added to DBL_MAX... */
  checkRangeF(LDBL_MAX, LDBL_MIN, LDBL_MAX-dtol, 0.0, CONTRACTS_TRUE);
  /* Note DBL_MAX gets lost when added to DBL_MAX... */
  checkRangeF(LDBL_MAX, LDBL_MIN, LDBL_MAX-DBL_MAX, 0.0, CONTRACTS_TRUE);
  checkRangeF(LDBL_MAX, LDBL_MIN, 0.9*LDBL_MAX, 0.0, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceRange */


/**
 * Executes all checkMaxF cases.
 */
void
checkPceMaxF()
{
  /* Should only need to use long double values at this point */
  checkMaxF(LDBL_MIN, LDBL_MAX, LDBL_MAX, CONTRACTS_TRUE);
  checkMaxF(LDBL_MAX, LDBL_MAX, LDBL_MAX, CONTRACTS_TRUE);
  checkMaxF(LDBL_MAX, LDBL_MIN, LDBL_MAX, CONTRACTS_TRUE);
  checkMaxF(LDBL_MIN, LDBL_MAX, LDBL_MIN, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceMaxF */


/**
 * Executes all checkMaxI cases.
 */
void
checkPceMaxI()
{
  /* Should only need to use long long values at this point */
  checkMaxI(LLONG_MIN, LLONG_MAX, LLONG_MAX, CONTRACTS_TRUE);
  checkMaxI(LLONG_MIN, LLONG_MIN, LLONG_MIN, CONTRACTS_TRUE);
  checkMaxI(LLONG_MAX, LLONG_MIN, LLONG_MAX, CONTRACTS_TRUE);
  checkMaxI(LLONG_MIN, LLONG_MAX, LLONG_MIN, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceMaxI */


/**
 * Executes all checkMinI cases.
 */
void
checkPceMinI()
{
  /* Should only need to use long long values at this point */
  checkMinI(LLONG_MIN, LLONG_MAX, LLONG_MIN, CONTRACTS_TRUE);
  checkMinI(LLONG_MIN, LLONG_MIN, LLONG_MIN, CONTRACTS_TRUE);
  checkMinI(LLONG_MAX, LLONG_MIN, LLONG_MIN, CONTRACTS_TRUE);
  checkMinI(LLONG_MIN, LLONG_MAX, LLONG_MAX, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceMinI */


/**
 * Executes all checkMinF cases.
 */
void
checkPceMinF()
{
  /* Should only need to use long long values at this point */
  checkMinF(LDBL_MIN, LDBL_MAX, LDBL_MIN, CONTRACTS_TRUE);
  checkMinF(LDBL_MAX, LDBL_MAX, LDBL_MAX, CONTRACTS_TRUE);
  checkMinF(LDBL_MAX, LDBL_MIN, LDBL_MIN, CONTRACTS_TRUE);
  checkMinF(LDBL_MIN, LDBL_MAX, LDBL_MAX, CONTRACTS_FALSE);

  printf("\n");
} /* checkPceMinF */


/**
 * Executes all checkNearEqualLD cases.
 */
void
checkPceNearEqual()
{
  /* Use long double values */
  long double ldtol = 0.0000001L;
  checkNearEqualLD(LDBL_MIN, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MIN+ldtol, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MIN+(2.0L*ldtol), LDBL_MIN, ldtol, CONTRACTS_FALSE);
  checkNearEqualLD(LDBL_MAX/(1.0L+ldtol), LDBL_MAX, ldtol, CONTRACTS_FALSE);
  checkNearEqualLD(LDBL_MAX-ldtol, LDBL_MAX, ldtol, CONTRACTS_TRUE);
  checkNearEqualLD(LDBL_MAX, LDBL_MAX, ldtol, CONTRACTS_TRUE);

  printf("\n");

  /* Use float values */
  float ftol = 0.0000001f;
  checkNearEqualLD(FLT_MIN, FLT_MIN+ftol, ftol, CONTRACTS_TRUE);
  checkNearEqualLD(FLT_MIN, FLT_MIN+(2.0f*ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualLD(FLT_MAX, FLT_MAX/(1.0f+ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualLD(FLT_MAX, FLT_MAX-ftol, ftol, CONTRACTS_TRUE);

  printf("\n");

  /* Use double values */
  double dtol = 0.0000001;
  checkNearEqualLD(DBL_MIN, DBL_MIN+dtol, dtol, CONTRACTS_TRUE);
  checkNearEqualLD(DBL_MIN, DBL_MIN+(2.0*dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualLD(DBL_MAX, DBL_MAX/(1.0+dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualLD(DBL_MAX, DBL_MAX-dtol, dtol, CONTRACTS_TRUE);

  printf("\n");
} /* checkPceNearEqual */



/**
 * Test driver.  
 */
int
main(int argc, char **argv)
{
  printf("\nRunning testExpressionRoutines tests...\n");

  /* pce_all_null checks */
  checkPceAllNullChars();
  checkPceAllNullInts();
  checkPceAllNullLDs();

  /* pce_any_null checks */
  checkPceAnyNullChars();
  checkPceAnyNullInts();
  checkPceAnyNullLDs();

  /* pce_in_range checks */
  checkPceInRange();

  /* pce_range checks */
  checkPceRange();

  /* pce_max checks */
  checkPceMaxI();
  checkPceMaxF();

  /* pce_min checks */
  checkPceMinI();
  checkPceMinF();

  /* pce_near_equal checks */
  checkPceNearEqual();

  /* Wrap up by summarizing results */
  printf("\n..%ld passed out of %ld cases\n", g_numOkay, g_numTests);
  printf("\n\nTEST SUITE %s\n", (g_numOkay==g_numTests) ? "PASSED" : "FAILED");

  return 0;
} /* main */ 
