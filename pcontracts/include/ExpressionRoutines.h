/**
 * \internal
 * File:           ExpressionRoutines.h
 * Author:         T. Dahlgren
 * Created:        2013 October 8
 * Last Modified:  2015 March 31
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
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_char(
  /* in */ const char*  arr,
  /* in */ const char*  rel,
  /* in */ const char   val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_double(
  /* in */ double*      arr,
  /* in */ const char*  rel,
  /* in */ double       val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_float(
  /* in */ float*       arr,
  /* in */ const char*  rel,
  /* in */ float        val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_int(
  /* in */ int*         arr,
  /* in */ const char*  rel,
  /* in */ int          val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_int64(
  /* in */ int64_t*     arr,
  /* in */ const char*  rel,
  /* in */ int64_t      val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_long(
  /* in */ long*        arr,
  /* in */ const char*  rel,
  /* in */ long         val,
  /* in */ int64_t      num);


/**
 * Determine if all (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_all_longdouble(
  /* in */ long double* arr,
  /* in */ const char*  rel,
  /* in */ long double  val,
  /* in */ int64_t      num);


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
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_char(
  /* in */ const char*  arr,
  /* in */ const char*  rel,
  /* in */ const char   val,
  /* in */ int64_t      num);



/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_double(
  /* in */ double*      arr,
  /* in */ const char*  rel,
  /* in */ double       val,
  /* in */ int64_t      num);


/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_float(
  /* in */ float*       arr,
  /* in */ const char*  rel,
  /* in */ float        val,
  /* in */ int64_t      num);


/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_int(
  /* in */ int*         arr,
  /* in */ const char*  rel,
  /* in */ int          val,
  /* in */ int64_t      num);


/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_int64(
  /* in */ int64_t*     arr,
  /* in */ const char*  rel,
  /* in */ int64_t      val,
  /* in */ int64_t      num);


/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_long(
  /* in */ long*        arr,
  /* in */ const char*  rel,
  /* in */ long         val,
  /* in */ int64_t      num);


/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
CONTRACTS_BOOL
pce_any_longdouble(
  /* in */ long double* arr,
  /* in */ const char*  rel,
  /* in */ long double  val,
  /* in */ int64_t      num);


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
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_char(
  /* in */ const char*  arr,
  /* in */ const char*  rel,
  /* in */ const char   val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_double(
  /* in */ double*      arr,
  /* in */ const char*  rel,
  /* in */ double       val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_float(
  /* in */ float*       arr,
  /* in */ const char*  rel,
  /* in */ float        val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_int(
  /* in */ int*         arr,
  /* in */ const char*  rel,
  /* in */ int          val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_int64(
  /* in */ int64_t*     arr,
  /* in */ const char*  rel,
  /* in */ int64_t      val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_long(
  /* in */ long*        arr,
  /* in */ const char*  rel,
  /* in */ long         val,
  /* in */ int64_t      num);


/**
 * Determine the number (of the first) num entries that have the specified 
 * relation to the value.
 *
 * @param[in] arr  The array variable.
 * @param[in] rel  The binary relationship operator (as a string).
 * @param[in] val  The value to be compared.
 * @param[in] num  The length or number of entries in the array.
 *
 * @return    Returns true if all are so related; otherwise, returns false.
 */
int64_t
pce_count_longdouble(
  /* in */ long double* arr,
  /* in */ const char*  rel,
  /* in */ long double  val,
  /* in */ int64_t      num);


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
