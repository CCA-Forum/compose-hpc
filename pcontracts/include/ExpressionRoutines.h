/**
 * \internal
 * File:           ExpressionRoutines.h
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2013 October 8
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
 **********************************************************************
 * PUBLIC ROUTINES
 **********************************************************************
 */


/**
 * \publicsection
 *
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
 * Determine if the provided variable is within the tolerance of the specified
 * value.
 *
 * @param[in] var    The variable whose value is being checked.
 * @param[in] value  The target equivalent value.
 * @param[in] tol    The allowable tolerance for the value range.
 *
 * @return    Returns true if @a var is in range; false otherwise.
 */
CONTRACTS_BOOL
pce_near_equal(
  /* in */ long double var,
  /* in */ long double value,
  /* in */ long double tol);

#endif /* Expression_Routines_h */
