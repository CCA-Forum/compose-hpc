/**
 * \internal
 * File:           ExpressionRoutines.h
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2015 February 5
 * \endinternal
 *
 * @file
 * @brief 
 * Interface contract expression (helper) routines.
 *
 * @htmlinclude copyright.html
 */


#ifndef Expression_Routines_h
#define Expression_Routines_h

#include <sys/types.h>

#include "contracts.h"


/*
 **********************************************************************
 * PRIVATE ROUTINES
 **********************************************************************
 */

/**
 * \privatesection
 */



/*
 *
 **********************************************************************
 * PUBLIC ROUTINES
 **********************************************************************
 */
/**
 * \publicsection
 */

#define pce_max(A, B) (((A) > (B)) ? (A) : (B))
#define pce_min(A, B) (((A) < (B)) ? (A) : (B))

/**
 * Determine if all (of the first) num entries are NULL.
 * 
 * @param[in] arr  The array of addresses variable.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are NULL; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_null(
  /* in */ void**  arr, 
  /* in */ int64_t num);


/**
 * Determine if any (of the first) num entries are NULL.
 *
 * @param[in] arr  The array of addresses variable.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are NULL; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_null(
  /* in */ void**  arr,
  /* in */ int64_t num);


/**
 * Determine if the provided variable is in the specified range.
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
  /* in */ int64_t maxvalue);


/**
 * Determine if the provided variable is within the tolerance of the
 * specified range.
 *
 * @param[in] var       The variable whose value is being checked.
 * @param[in] minvalue  The lowest value @a var can take on in the range.
 * @param[in] maxvalue  The highest value @a var can take on in the range.
 * @param[in] tol       The allowed tolerance for the min and max values.
 *
 * @return    Returns true if @a var is within tolerance of the range; false 
 * otherwise.
 */
CONTRACTS_BOOL
pce_range(
  /* in */ long double var,
  /* in */ long double minvalue,
  /* in */ long double maxvalue,
  /* in */ long double tol);


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
  /* in */ long double tol);


#endif /* Expression_Routines_h */
