/**
 * \internal
 * File:           testExpressionRoutines.c
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2015 March 17
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
 * Checks pce_all_double for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllD(
  /* in */ double*        arr,
  /* in */ const char*    rel,
  /* in */ double         val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_double(%s, %s, %g, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_all_double(arr, rel, val, num) == result) 
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
} /* checkAllD */


/**
 * Executes suite of tests for check_all_double.
 */
void
checkPceAllD()
{
  int i, sz = 20;
  const char *desc = "darr";
  const char *desc5 = "darr.5";
  double *darr = (double *)malloc((size_t)sz * sizeof(double));

  for (i = 0; i < sz; i++) darr[i] = DBL_MIN;
  checkAllD(darr, "==", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllD(darr, "<", DBL_MIN, sz, desc, CONTRACTS_FALSE);
  checkAllD(darr, "<=", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllD(darr, ">", DBL_MIN, sz, desc, CONTRACTS_FALSE);
  checkAllD(darr, ">=", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllD(darr, "!=", DBL_MIN, sz, desc, CONTRACTS_FALSE);

  darr[sz-1] = DBL_MAX*.5;
  checkAllD(darr, "==", DBL_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAllD(darr, "<", DBL_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAllD(darr, "<=", DBL_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAllD(darr, ">", DBL_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAllD(darr, ">=", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAllD(darr, "!=", DBL_MIN, sz, desc5, CONTRACTS_FALSE);

  printf("\n");

  free((void*)darr);
} /* checkPceAllD */


/**
 * Checks pce_all_int for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllInt(
  /* in */ int*           arr,
  /* in */ const char*    rel,
  /* in */ int            val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_int(%s, %s, %d, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_all_int(arr, rel, val, num) == result) 
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
} /* checkAllInt */


/**
 * Executes suite of tests for check_all_int.
 */
void
checkPceAllInt()
{
  int i, sz = 20;
  const char *desc = "iarr";
  const char *desc0 = "iarr0";
  const char *desc0M = "iarr0M";
  int *iarr = (int *)malloc((size_t)sz * sizeof(int));

  for (i = 0; i < sz; i++) iarr[i] = INT_MAX;
  checkAllInt(iarr, "==", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllInt(iarr, "<", INT_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllInt(iarr, "<=", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllInt(iarr, ">", INT_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllInt(iarr, ">=", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllInt(iarr, "!=", INT_MAX, sz, desc, CONTRACTS_FALSE);

  iarr[sz-1] = 0;
  checkAllInt(iarr, "==", INT_MAX, sz, desc0, CONTRACTS_FALSE);
  checkAllInt(iarr, "<", INT_MAX, sz, desc0, CONTRACTS_FALSE);
  checkAllInt(iarr, "<=", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAllInt(iarr, ">", INT_MAX, sz, desc0, CONTRACTS_FALSE);
  checkAllInt(iarr, ">=", INT_MAX, sz, desc0, CONTRACTS_FALSE);
  checkAllInt(iarr, "!=", INT_MAX, sz, desc0, CONTRACTS_FALSE);

  iarr[0] = 0;
  iarr[sz-1] = INT_MAX;
  checkAllInt(iarr, "==", INT_MAX, sz, desc0M, CONTRACTS_FALSE);
  checkAllInt(iarr, "<", INT_MAX, sz, desc0M, CONTRACTS_FALSE);
  checkAllInt(iarr, "<=", INT_MAX, sz, desc0M, CONTRACTS_TRUE);
  checkAllInt(iarr, ">", INT_MAX, sz, desc0M, CONTRACTS_FALSE);
  checkAllInt(iarr, ">=", INT_MAX, sz, desc0M, CONTRACTS_FALSE);
  checkAllInt(iarr, "!=", INT_MAX, sz, desc0M, CONTRACTS_FALSE);

  printf("\n");

  free((void*)iarr);
} /* checkPceAllInt */



/**
 * Checks pce_all_int64 for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllI64(
  /* in */ int64_t*       arr,
  /* in */ const char*    rel,
  /* in */ int64_t        val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_int64(%s, %s, %lld, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_all_int64(arr, rel, val, num) == result) 
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
} /* checkAllI64 */


/**
 * Executes suite of tests for check_all_i64.
 */
void
checkPceAllI64()
{
  int i, sz = 20;
  const char *desc = "i64arr";
  const char *desc0 = "i64arr0";
  int64_t *i64arr = (int64_t *)malloc((size_t)sz * sizeof(int64_t));

  for (i = 0; i < sz; i++) i64arr[i] = LLONG_MIN;
  checkAllI64(i64arr, "==", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllI64(i64arr, "<", LLONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAllI64(i64arr, "<=", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllI64(i64arr, ">", LLONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAllI64(i64arr, ">=", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAllI64(i64arr, "!=", LLONG_MIN, sz, desc, CONTRACTS_FALSE);

  i64arr[sz-1] = (int64_t)0.0;
  checkAllI64(i64arr, "==", LLONG_MIN, sz, desc0, CONTRACTS_FALSE);
  checkAllI64(i64arr, "<", LLONG_MIN, sz, desc0, CONTRACTS_FALSE);
  checkAllI64(i64arr, "<=", LLONG_MIN, sz, desc0, CONTRACTS_FALSE);
  checkAllI64(i64arr, ">", LLONG_MIN, sz, desc0, CONTRACTS_FALSE);
  checkAllI64(i64arr, ">=", LLONG_MIN, sz, desc0, CONTRACTS_TRUE);
  checkAllI64(i64arr, "!=", LLONG_MIN, sz, desc0, CONTRACTS_FALSE);

  printf("\n");

  free((void*)i64arr);
} /* checkPceAllI64 */



/**
 * Checks pce_all_long for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllL(
  /* in */ long*          arr,
  /* in */ const char*    rel,
  /* in */ long           val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_long(%s, %s, %ld, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_all_long(arr, rel, val, num) == result) 
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
} /* checkAllL */


/**
 * Executes suite of tests for check_all_long.
 */
void
checkPceAllL()
{
  int i, sz = 20;
  const char *desc = "larr";
  const char *desc9 = "larr.9";
  long *larr = (long *)malloc((size_t)sz * sizeof(long));

  for (i = 0; i < sz; i++) larr[i] = LONG_MAX;
  checkAllL(larr, "==", LONG_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllL(larr, "<", LONG_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllL(larr, "<=", LONG_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllL(larr, ">", LONG_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllL(larr, ">=", LONG_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllL(larr, "!=", LONG_MAX, sz, desc, CONTRACTS_FALSE);

  larr[sz-1] = LONG_MAX*0.9;
  checkAllL(larr, "==", LONG_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllL(larr, "<", LONG_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllL(larr, "<=", LONG_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAllL(larr, ">", LONG_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllL(larr, ">=", LONG_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllL(larr, "!=", LONG_MAX, sz, desc9, CONTRACTS_FALSE);

  printf("\n");

  free((void*)larr);
} /* checkPceAllL */



/**
 * Checks pce_all_longdouble for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAllLD(
  /* in */ long double*   arr,
  /* in */ const char*    rel,
  /* in */ long double    val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_all_longdouble(%s, %s, %Lg, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_all_longdouble(arr, rel, val, num) == result) 
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
} /* checkAllLD */


/**
 * Executes suite of tests for check_all_longdouble.
 */
void
checkPceAllLD()
{
  int i, sz = 20;
  const char *desc = "ldarr";
  const char *desc9 = "ldarr.9";
  long double *ldarr = (long double *)malloc((size_t)sz * sizeof(long double));

  for (i = 0; i < sz; i++) ldarr[i] = LDBL_MAX;
  checkAllLD(ldarr, "==", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllLD(ldarr, "<", LDBL_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllLD(ldarr, "<=", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllLD(ldarr, ">", LDBL_MAX, sz, desc, CONTRACTS_FALSE);
  checkAllLD(ldarr, ">=", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAllLD(ldarr, "!=", LDBL_MAX, sz, desc, CONTRACTS_FALSE);

  ldarr[sz-1] = LDBL_MAX*0.9;
  checkAllLD(ldarr, "==", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllLD(ldarr, "<", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllLD(ldarr, "<=", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAllLD(ldarr, ">", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllLD(ldarr, ">=", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAllLD(ldarr, "!=", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);

  printf("\n");

  free((void*)ldarr);
} /* checkPceAllLD */



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
  printf("%3ld. pce_all_null(%s, %lld) == %d:  ",
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
 * Checks pce_any_double for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyD(
  /* in */ double*        arr,
  /* in */ const char*    rel,
  /* in */ double         val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_double(%s, %s, %g, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_any_double(arr, rel, val, num) == result) 
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
} /* checkAnyD */


/**
 * Executes a suite of checks for pce_any_double.
 */
void
checkPceAnyD()
{
  int i, sz = 20;
  const char *desc = "darr";
  const char *desc5 = "darr.5";
  double *darr = (double *)malloc((size_t)sz * sizeof(double));

  for (i = 0; i < sz; i++) darr[i] = DBL_MIN;
  checkAnyD(darr, "==", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyD(darr, "<", DBL_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyD(darr, "<=", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyD(darr, ">", DBL_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyD(darr, ">=", DBL_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyD(darr, "!=", DBL_MIN, sz, desc, CONTRACTS_FALSE);

  darr[sz-1] = DBL_MAX*0.5;
  checkAnyD(darr, "==", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyD(darr, "<", DBL_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAnyD(darr, "<=", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyD(darr, ">", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyD(darr, ">=", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyD(darr, "!=", DBL_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyD(darr, "==", darr[sz-1], sz, desc5, CONTRACTS_TRUE);

  printf("\n");

  free((void*)darr);
} /* checkPceAnyD */



/**
 * Checks pce_any_int for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyInt(
  /* in */ int*           arr,
  /* in */ const char*    rel,
  /* in */ int            val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_int(%s, %s, %d, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_any_int(arr, rel, val, num) == result) 
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
} /* checkAnyInt */


/**
 * Executes a suite of checks for pce_any_int.
 */
void
checkPceAnyInt()
{
  int i, sz = 20;
  const char *desc = "iarr";
  const char *desc0 = "iarr0";
  int *iarr = (int *)malloc((size_t)sz * sizeof(int));

  for (i = 0; i < sz; i++) iarr[i] = INT_MAX;
  checkAnyInt(iarr, "==", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyInt(iarr, "<", INT_MAX, sz, desc, CONTRACTS_FALSE);
  checkAnyInt(iarr, "<=", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyInt(iarr, ">", INT_MAX, sz, desc, CONTRACTS_FALSE);
  checkAnyInt(iarr, ">=", INT_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyInt(iarr, "!=", INT_MAX, sz, desc, CONTRACTS_FALSE);

  iarr[sz-1] = 0;
  checkAnyInt(iarr, "==", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAnyInt(iarr, "<", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAnyInt(iarr, "<=", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAnyInt(iarr, ">", INT_MAX, sz, desc0, CONTRACTS_FALSE);
  checkAnyInt(iarr, ">=", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAnyInt(iarr, "!=", INT_MAX, sz, desc0, CONTRACTS_TRUE);
  checkAnyInt(iarr, "==", 0, sz, desc0, CONTRACTS_TRUE);

  printf("\n");

  free((void*)iarr);
} /* checkPceAnyInt */



/**
 * Checks pce_any_int for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyInt64(
  /* in */ int64_t*       arr,
  /* in */ const char*    rel,
  /* in */ int64_t        val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_int64(%s, %s, %lld, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_any_int64(arr, rel, val, num) == result) 
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
} /* checkAnyInt64 */


/**
 * Executes a suite of checks for pce_any_int64.
 */
void
checkPceAnyI64()
{
  int i, sz = 20;
  const char *desc = "i64arr";
  const char *desc5 = "i64arr.5";
  int64_t *i64arr = (int64_t *)malloc((size_t)sz * sizeof(int64_t));

  for (i = 0; i < sz; i++) i64arr[i] = LLONG_MIN;
  checkAnyInt64(i64arr, "==", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, "<", LLONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyInt64(i64arr, "<=", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, ">", LLONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyInt64(i64arr, ">=", LLONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, "!=", LLONG_MIN, sz, desc, CONTRACTS_FALSE);

  i64arr[sz-1] = LLONG_MAX*0.5;
  checkAnyInt64(i64arr, "==", LLONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, "<", LLONG_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAnyInt64(i64arr, "<=", LLONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, ">", LLONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, ">=", LLONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, "!=", LLONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyInt64(i64arr, "==", i64arr[sz-1], sz, desc5, CONTRACTS_TRUE);

  printf("\n");

  free((void*)i64arr);
} /* checkPceAnyI64 */



/**
 * Checks pce_any_long for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyL(
  /* in */ long*          arr,
  /* in */ const char*    rel,
  /* in */ long           val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_long(%s, %s, %g, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_any_long(arr, rel, val, num) == result) 
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
} /* checkAnyL */


/**
 * Executes a suite of checks for pce_any_long.
 */
void
checkPceAnyL()
{
  int i, sz = 20;
  const char *desc = "larr";
  const char *desc5 = "larr.5";
  long *larr = (long *)malloc((size_t)sz * sizeof(long));

  for (i = 0; i < sz; i++) larr[i] = LONG_MIN;
  checkAnyL(larr, "==", LONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyL(larr, "<", LONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyL(larr, "<=", LONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyL(larr, ">", LONG_MIN, sz, desc, CONTRACTS_FALSE);
  checkAnyL(larr, ">=", LONG_MIN, sz, desc, CONTRACTS_TRUE);
  checkAnyL(larr, "!=", LONG_MIN, sz, desc, CONTRACTS_FALSE);

  larr[sz-1] = LONG_MAX*0.5;
  checkAnyL(larr, "==", LONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyL(larr, "<", LONG_MIN, sz, desc5, CONTRACTS_FALSE);
  checkAnyL(larr, "<=", LONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyL(larr, ">", LONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyL(larr, ">=", LONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyL(larr, "!=", LONG_MIN, sz, desc5, CONTRACTS_TRUE);
  checkAnyL(larr, "==", larr[sz-1], sz, desc5, CONTRACTS_TRUE);

  printf("\n");

  free((void*)larr);
} /* checkPceAnyL */



/**
 * Checks pce_any_longdouble for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkAnyLD(
  /* in */ long double*   arr,
  /* in */ const char*    rel,
  /* in */ long double    val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. pce_any_longdouble(%s, %s, %Lg, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_any_longdouble(arr, rel, val, num) == result) 
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
} /* checkAnyLD */


/**
 * Executes a suite of checks for pce_any_longdouble.
 */
void
checkPceAnyLD()
{
  int i, sz = 20;
  const char *desc = "d64arr";
  const char *desc9 = "d64arr.9";
  long double *d64arr = (long double *)malloc((size_t)sz * sizeof(long double));

  for (i = 0; i < sz; i++) d64arr[i] = LDBL_MAX;
  checkAnyLD(d64arr, "==", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "<", LDBL_MAX, sz, desc, CONTRACTS_FALSE);
  checkAnyLD(d64arr, "<=", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyLD(d64arr, ">", LDBL_MAX, sz, desc, CONTRACTS_FALSE);
  checkAnyLD(d64arr, ">=", LDBL_MAX, sz, desc, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "!=", LDBL_MAX, sz, desc, CONTRACTS_FALSE);

  d64arr[sz-1] = LDBL_MAX*0.9;
  checkAnyLD(d64arr, "==", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "<", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "<=", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAnyLD(d64arr, ">", LDBL_MAX, sz, desc9, CONTRACTS_FALSE);
  checkAnyLD(d64arr, ">=", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "!=", LDBL_MAX, sz, desc9, CONTRACTS_TRUE);
  checkAnyLD(d64arr, "==", d64arr[sz-1], sz, desc9, CONTRACTS_TRUE);

  printf("\n");

  free((void*)d64arr);
} /* checkPceAnyLD */



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
  printf("%3ld. pce_any_null(%s, %lld) == %d:  ",
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
 * Checks pce_count_double for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkCountD(
  /* in */ double*        arr,
  /* in */ const char*    rel,
  /* in */ double         val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ int64_t        result)
{
  g_numTests += 1;
  printf("%3ld. pce_count_double(%s, %s, %g, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_count_double(arr, rel, val, num) == result) 
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
} /* checkCountD */


/**
 * Executes a suite of checks for pce_count_double.
 */
void
checkPceCountD()
{
  int i, sz = 20;
  const char *desc = "darr";
  const char *desc5 = "darr.5";
  double *darr = (double *)malloc((size_t)sz * sizeof(double));

  for (i = 0; i < sz; i++) darr[i] = DBL_MIN;
  checkCountD(darr, "==", DBL_MIN, sz, desc, sz);
  checkCountD(darr, "<", DBL_MIN, sz, desc, 0);
  checkCountD(darr, "<=", DBL_MIN, sz, desc, sz);
  checkCountD(darr, ">", DBL_MIN, sz, desc, 0);
  checkCountD(darr, ">=", DBL_MIN, sz, desc, sz);
  checkCountD(darr, "!=", DBL_MIN, sz, desc, 0);

  darr[sz-1] = DBL_MAX*0.5;
  checkCountD(darr, "==", DBL_MIN, sz, desc5, sz-1);
  checkCountD(darr, "<", DBL_MIN, sz, desc5, 0);
  checkCountD(darr, "<=", DBL_MIN, sz, desc5, sz-1);
  checkCountD(darr, ">", DBL_MIN, sz, desc5, 1);
  checkCountD(darr, ">=", DBL_MIN, sz, desc5, sz);
  checkCountD(darr, "!=", DBL_MIN, sz, desc5, 1);
  checkCountD(darr, "==", darr[sz-1], sz, desc5, 1);

  printf("\n");

  free((void*)darr);
} /* checkPceCountD */



/**
 * Checks pce_count_int for an expected result.
 *
 * @param[in] arr     The array whose (pointer) contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkCountInt(
  /* in */ int*           arr,
  /* in */ const char*    rel,
  /* in */ int            val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ int64_t        result)
{
  g_numTests += 1;
  printf("%3ld. pce_count_int(%s, %s, %d, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_count_int(arr, rel, val, num) == result) 
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
} /* checkCountInt */


/**
 * Executes a suite of checks for pce_count_int.
 */
void
checkPceCountInt()
{
  int i, sz = 20;
  const char *desc = "iarr";
  const char *desc0 = "iarr0";
  int *iarr = (int *)malloc((size_t)sz * sizeof(int));

  for (i = 0; i < sz; i++) iarr[i] = INT_MAX;
  checkCountInt(iarr, "==", INT_MAX, sz, desc, sz);
  checkCountInt(iarr, "<", INT_MAX, sz, desc, 0);
  checkCountInt(iarr, "<=", INT_MAX, sz, desc, sz);
  checkCountInt(iarr, ">", INT_MAX, sz, desc, 0);
  checkCountInt(iarr, ">=", INT_MAX, sz, desc, sz);
  checkCountInt(iarr, "!=", INT_MAX, sz, desc, 0);

  iarr[sz-1] = 0;
  checkCountInt(iarr, "==", INT_MAX, sz, desc0, sz-1);
  checkCountInt(iarr, "<", INT_MAX, sz, desc0, 1);
  checkCountInt(iarr, "<=", INT_MAX, sz, desc0, sz);
  checkCountInt(iarr, ">", INT_MAX, sz, desc0, 0);
  checkCountInt(iarr, ">=", INT_MAX, sz, desc0, sz-1);
  checkCountInt(iarr, "!=", INT_MAX, sz, desc0, 1);
  checkCountInt(iarr, "==", 0, sz, desc0, 1);

  printf("\n");

  free((void*)iarr);
} /* checkPceCountInt */



/**
 * Checks pce_count_int64 for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkCountI64(
  /* in */ int64_t*       arr,
  /* in */ const char*    rel,
  /* in */ int64_t        val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ int64_t        result)
{
  g_numTests += 1;
  printf("%3ld. pce_count_int64(%s, %s, %lld, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_count_int64(arr, rel, val, num) == result) 
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
} /* checkCountI64 */


/**
 * Executes a suite of checks for pce_count_int64.
 */
void
checkPceCountI64()
{
  int i, sz = 20;
  const char *desc = "i64arr";
  const char *desc5 = "i64arr.5";
  int64_t *i64arr = (int64_t *)malloc((size_t)sz * sizeof(int64_t));

  for (i = 0; i < sz; i++) i64arr[i] = LLONG_MIN;
  checkCountI64(i64arr, "==", LLONG_MIN, sz, desc, sz);
  checkCountI64(i64arr, "<", LLONG_MIN, sz, desc, 0);
  checkCountI64(i64arr, "<=", LLONG_MIN, sz, desc, sz);
  checkCountI64(i64arr, ">", LLONG_MIN, sz, desc, 0);
  checkCountI64(i64arr, ">=", LLONG_MIN, sz, desc, sz);
  checkCountI64(i64arr, "!=", LLONG_MIN, sz, desc, 0);

  i64arr[sz-1] = LLONG_MAX*0.5;
  checkCountI64(i64arr, "==", LLONG_MIN, sz, desc5, sz-1);
  checkCountI64(i64arr, "<", LLONG_MIN, sz, desc5, 0);
  checkCountI64(i64arr, "<=", LLONG_MIN, sz, desc5, sz-1);
  checkCountI64(i64arr, ">", LLONG_MIN, sz, desc5, 1);
  checkCountI64(i64arr, ">=", LLONG_MIN, sz, desc5, sz);
  checkCountI64(i64arr, "!=", LLONG_MIN, sz, desc5, 1);
  checkCountI64(i64arr, "==", i64arr[sz-1], sz, desc5, 1);

  printf("\n");

  free((void*)i64arr);
} /* checkPceCountI64 */



/**
 * Checks pce_count_long for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkCountL(
  /* in */ long*          arr,
  /* in */ const char*    rel,
  /* in */ long           val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ int64_t        result)
{
  g_numTests += 1;
  printf("%3ld. pce_count_long(%s, %s, %ld, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_count_long(arr, rel, val, num) == result) 
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
} /* checkCountL */


/**
 * Executes a suite of checks for pce_count_long.
 */
void
checkPceCountL()
{
  int i, sz = 20;
  const char *desc = "ldarr";
  const char *desc9 = "ldarr.9";
  long *larr = (long *)malloc((size_t)sz * sizeof(long));

  for (i = 0; i < sz; i++) larr[i] = LONG_MAX;
  checkCountL(larr, "==", LONG_MAX, sz, desc, sz);
  checkCountL(larr, "<", LONG_MAX, sz, desc, 0);
  checkCountL(larr, "<=", LONG_MAX, sz, desc, sz);
  checkCountL(larr, ">", LONG_MAX, sz, desc, 0);
  checkCountL(larr, ">=", LONG_MAX, sz, desc, sz);
  checkCountL(larr, "!=", LONG_MAX, sz, desc, 0);

  larr[sz-1] = LONG_MAX*0.9;
  checkCountL(larr, "==", LONG_MAX, sz, desc9, sz-1);
  checkCountL(larr, "<", LONG_MAX, sz, desc9, 1);
  checkCountL(larr, "<=", LONG_MAX, sz, desc9, sz);
  checkCountL(larr, ">", LONG_MAX, sz, desc9, 0);
  checkCountL(larr, ">=", LONG_MAX, sz, desc9, sz-1);
  checkCountL(larr, "!=", LONG_MAX, sz, desc9, 1);
  checkCountL(larr, "==", larr[sz-1], sz, desc9, 1);

  printf("\n");

  free((void*)larr);
} /* checkPceCountL */



/**
 * Checks pce_count_longdouble for an expected result.
 *
 * @param[in] arr     The array whose contents are being checked.
 * @param[in] rel     The binary relationship operator (as a string).
 * @param[in] val     The value to be compared.
 * @param[in] num     The number of entries in the array (to check).
 * @param[in] desc    The character description of the array check.
 * @param[in] result  The expected result from the test.
 */ 
void
checkCountLD(
  /* in */ long double*   arr,
  /* in */ const char*    rel,
  /* in */ long double    val,
  /* in */ int64_t        num,
  /* in */ const char*    desc,
  /* in */ int64_t        result)
{
  g_numTests += 1;
  printf("%3ld. pce_count_longdouble(%s, %s, %Lg, %lld) == %d:  ",
    g_numTests, desc, rel, val, num, result);

  if (pce_count_longdouble(arr, rel, val, num) == result) 
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
} /* checkCountLD */


/**
 * Executes a suite of checks for pce_count_longdouble.
 */
void
checkPceCountLD()
{
  int i, sz = 20;
  const char *desc = "ldarr";
  const char *desc9 = "ldarr.9";
  long double *ldarr = (long double *)malloc((size_t)sz * sizeof(long double));

  for (i = 0; i < sz; i++) ldarr[i] = LDBL_MAX;
  checkCountLD(ldarr, "==", LDBL_MAX, sz, desc, sz);
  checkCountLD(ldarr, "<", LDBL_MAX, sz, desc, 0);
  checkCountLD(ldarr, "<=", LDBL_MAX, sz, desc, sz);
  checkCountLD(ldarr, ">", LDBL_MAX, sz, desc, 0);
  checkCountLD(ldarr, ">=", LDBL_MAX, sz, desc, sz);
  checkCountLD(ldarr, "!=", LDBL_MAX, sz, desc, 0);

  ldarr[sz-1] = LDBL_MAX*0.9;
  checkCountLD(ldarr, "==", LDBL_MAX, sz, desc9, sz-1);
  checkCountLD(ldarr, "<", LDBL_MAX, sz, desc9, 1);
  checkCountLD(ldarr, "<=", LDBL_MAX, sz, desc9, sz);
  checkCountLD(ldarr, ">", LDBL_MAX, sz, desc9, 0);
  checkCountLD(ldarr, ">=", LDBL_MAX, sz, desc9, sz-1);
  checkCountLD(ldarr, "!=", LDBL_MAX, sz, desc9, 1);
  checkCountLD(ldarr, "==", ldarr[sz-1], sz, desc9, 1);

  printf("\n");

  free((void*)ldarr);
} /* checkPceCountLD */



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
  printf("%3ld. pce_in_range(%lld, %lld, %lld) == %d:  ",
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
  printf("%3ld. pce_range(%Lg, %Lg, %Lg, %Lg) == %d:  ",
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
  printf("%3ld. (checkMaxF(%Lg, %Lg) == %Lg) == %d:  ", g_numTests, a, b, 
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
  printf("%3ld. (checkMaxI(%lld, %lld) == %lld) == %d:  ", g_numTests, a, b, 
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
  printf("%3ld. (checkMinF(%Lg, %Lg) == %Lg) == %d:  ", g_numTests, a, b, 
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
  printf("%3ld. (checkMinI(%lld, %lld) == %lld) == %d:  ", g_numTests, a, b, 
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
checkNearEqualF(
  /* in */ long double    var,
  /* in */ long double    value,
  /* in */ long double    tol,
  /* in */ CONTRACTS_BOOL result)
{
  g_numTests += 1;
  printf("%3ld. checkNearEqualF(%Lg, %Lg, %Lg) == %d:  ",
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
} /* checkNearEqualF */


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
checkPceAllNullIs()
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
} /* checkAllNullIs */


/**
 * Executes checkAllNull cases for a long double array.
 */
void
checkPceAllNullFs()
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
} /* checkAllNullFs */


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
checkPceAnyNullIs()
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
} /* checkAnyNullIs */


/**
 * Executes checkAnyNull cases for a long double array.
 */
void
checkPceAnyNullFs()
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
} /* checkAnyNullFs */


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
 * Executes all checkNearEqualF cases.
 */
void
checkPceNearEqual()
{
  /* Use long double values */
  long double ldtol = 0.0000001L;
  checkNearEqualF(LDBL_MIN, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualF(LDBL_MIN+ldtol, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualF(LDBL_MIN+(2.0L*ldtol), LDBL_MIN, ldtol, CONTRACTS_FALSE);
  checkNearEqualF(LDBL_MAX/(1.0L+ldtol), LDBL_MAX, ldtol, CONTRACTS_FALSE);
  checkNearEqualF(LDBL_MAX-ldtol, LDBL_MAX, ldtol, CONTRACTS_TRUE);
  checkNearEqualF(LDBL_MAX, LDBL_MAX, ldtol, CONTRACTS_TRUE);

  printf("\n");

  /* Use float values */
  float ftol = 0.0000001f;
  checkNearEqualF(FLT_MIN, FLT_MIN+ftol, ftol, CONTRACTS_TRUE);
  checkNearEqualF(FLT_MIN, FLT_MIN+(2.0f*ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualF(FLT_MAX, FLT_MAX/(1.0f+ftol), ftol, CONTRACTS_FALSE);
  checkNearEqualF(FLT_MAX, FLT_MAX-ftol, ftol, CONTRACTS_TRUE);

  printf("\n");

  /* Use double values */
  double dtol = 0.0000001;
  checkNearEqualF(DBL_MIN, DBL_MIN+dtol, dtol, CONTRACTS_TRUE);
  checkNearEqualF(DBL_MIN, DBL_MIN+(2.0*dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualF(DBL_MAX, DBL_MAX/(1.0+dtol), dtol, CONTRACTS_FALSE);
  checkNearEqualF(DBL_MAX, DBL_MAX-dtol, dtol, CONTRACTS_TRUE);

  printf("\n");
} /* checkPceNearEqual */


/**
 * Executes all checkPceRels cases.
 */
void
checkPceRels()
{
  /* Use long double values */
  long double ldtol = 0.0000001L;
  checkNearEqualF(LDBL_MIN, LDBL_MIN, ldtol, CONTRACTS_TRUE);
  checkNearEqualF(LDBL_MIN+ldtol, LDBL_MIN, ldtol, CONTRACTS_TRUE);
} /* checkPceRels */



/**
 * Test driver.  
 */
int
main(int argc, char **argv)
{
  printf("\nRunning testExpressionRoutines tests...\n");

  /* pce_all checks */
  checkPceAllD();
  checkPceAllInt();
  checkPceAllI64();
  checkPceAllL();
  checkPceAllLD();

  /* pce_all_null checks */
  checkPceAllNullChars();
  checkPceAllNullIs();
  checkPceAllNullFs();

  /* pce_any checks */
  checkPceAnyD();
  checkPceAnyInt();
  checkPceAnyI64();
  checkPceAnyL();
  checkPceAnyLD();

  /* pce_any_null checks */
  checkPceAnyNullChars();
  checkPceAnyNullIs();
  checkPceAnyNullFs();

  /* pce_count checks */
  checkPceCountD();
  checkPceCountInt();
  checkPceCountI64();
  checkPceCountL();
  checkPceCountLD();

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
  printf("\n..%lld passed out of %ld cases\n", g_numOkay, g_numTests);
  printf("\n\nTEST SUITE %s\n", (g_numOkay==g_numTests) ? "PASSED" : "FAILED");

  return 0;
} /* main */ 
