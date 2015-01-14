/**
 * \internal
 * File:           ExpressionRoutines.c
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2015 January 13
 * \endinternal
 *
 * @file
 * @brief 
 * Interface contract expression (helper) routines.
 *
 * @htmlinclude copyright.html
 */


#include <math.h>
#include <sys/types.h>
//#include <stdio.h>
//#include <string.h>

#include "ExpressionRoutines.h"


/*
 **********************************************************************
 * PRIVATE ROUTINES
 **********************************************************************
 */

/**
 * \privatesection
 */



/*
 **********************************************************************
 * PUBLIC ROUTINES
 **********************************************************************
 */
/**
 * \publicsection
 */

/**
 * Determine if all (of the first) num entries are NULL.
 *
 * @param[in] arr  The array variable.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are NULL; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_null(
  /* in */ void**  arr, 
  /* in */ int64_t num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  if (num >= 0)
  {
      _are = CONTRACTS_TRUE;
      for (int64_t i = 0; i < num; i++) 
      {
          if (arr[i] != NULL)
          {
              _are = CONTRACTS_FALSE;
              break;
          }
      }
  }
  return _are;
}  /* pce_all_null */


/**
 * Determine if any (of the first) num entries are NULL.
 *
 * @param[in] arr  The array variable.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are NULL; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_null(
  /* in */ void**  arr, 
  /* in */ int64_t num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  if (num >= 0)
  {
      for (int64_t i = 0; i < num; i++) 
      {
          if (arr[i] == NULL)
          {
              _are = CONTRACTS_TRUE;
              break;
          }
      }
  }
  return _are;
}


/**
 *
 * Determine if the provided variable is in the specified inclusive range.
 *
 * @param[in] var       The variable whose value is being checked.
 * @param[in] minvalue  The lowest value @a var can take on in the range.
 * @param[in] maxvalue  The highest value @a var can take on in the range.
 *
 * @return    Returns true if @a var is in range; false otherwise.
 */
CONTRACTS_BOOL
pce_in_range(
  /* in */ int64_t var,
  /* in */ int64_t minvalue,
  /* in */ int64_t maxvalue)
{
  return (minvalue <= var) && (var <= maxvalue);
}  /* pce_in_range */


/**
 * Determine if the provided variable is within the tolerance of the specified
 * value.
 *
 * @param[in] var  The variable whose value is being checked.
 * @param[in] val  The target equivalent value.
 * @param[in] tol  The allowable tolerance for the value range.
 *
 * @return    Returns true if @a var is in range; false otherwise.
 */
CONTRACTS_BOOL
pce_near_equal(
  /* in */ long double var,
  /* in */ long double val,
  /* in */ long double tol)
{
  return (fabsl(var-val) <= tol);
}  /* pce_near_equal */
