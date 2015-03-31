/**
 * \internal
 * File:           ExpressionRoutines.c
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


#include <math.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>

#include "ExpressionRoutines.h"


/*
 **********************************************************************
 * PRIVATE ROUTINES
 **********************************************************************
 */

/**
 * \privatesection
 */
#define CONTRACTS_PCE_ALL(ARR, REL, VAL, NUM, I, BRES) { \
    BRES = CONTRACTS_FALSE; \
    if (NUM > 0) { \
        BRES = CONTRACTS_TRUE; \
        if (strcmp(REL, "==") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] != VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else if (strcmp(REL, "!=") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] == VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else if (strcmp(REL, "<=") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] > VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else if (strcmp(REL, ">=") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] < VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else if (strcmp(REL, "<") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] >= VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else if (strcmp(REL, ">") == 0) { \
            for (I=0; (I<NUM) && (BRES); I++) { \
                if (ARR[I] <= VAL) BRES = CONTRACTS_FALSE; \
            } \
        } else { \
            BRES = CONTRACTS_FALSE; \
        } \
    } \
}

#define CONTRACTS_PCE_ANY(ARR, REL, VAL, NUM, I, BRES) { \
    BRES = CONTRACTS_FALSE; \
    if (strcmp(REL, "==") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] == VAL) BRES = CONTRACTS_TRUE; \
        } \
    } else if (strcmp(REL, "!=") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] != VAL) BRES = CONTRACTS_TRUE; \
        } \
    } else if (strcmp(REL, "<=") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] <= VAL) BRES = CONTRACTS_TRUE; \
        } \
    } else if (strcmp(REL, ">=") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] >= VAL) BRES = CONTRACTS_TRUE; \
        } \
    } else if (strcmp(REL, "<") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] < VAL) BRES = CONTRACTS_TRUE; \
        } \
    } else if (strcmp(REL, ">") == 0) { \
        for (I=0; (I<NUM) && (!BRES); I++) { \
            if (ARR[I] > VAL) BRES = CONTRACTS_TRUE; \
        } \
    } \
}

#define CONTRACTS_PCE_COUNT(ARR, REL, VAL, NUM, I, ICNT) { \
    ICNT = 0; \
    if (strcmp(REL, "==") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] == VAL) (ICNT)++; \
        } \
    } else if (strcmp(REL, "!=") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] != VAL) (ICNT)++; \
        } \
    } else if (strcmp(REL, "<=") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] <= VAL) (ICNT)++; \
        } \
    } else if (strcmp(REL, ">=") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] >= VAL) (ICNT)++; \
        } \
    } else if (strcmp(REL, "<") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] < VAL) (ICNT)++; \
        } \
    } else if (strcmp(REL, ">") == 0) { \
        for (I=0; I<NUM; I++) { \
            if (ARR[I] > VAL) (ICNT)++; \
        } \
    } \
}



/*
 **********************************************************************
 * PUBLIC ROUTINES
 **********************************************************************
 */
/**
 * \publicsection
 */

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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_all_char */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_double */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_float */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_int */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_int64 */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_long */



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
  /* in */ long double*  arr,
  /* in */ const char*   rel,
  /* in */ long double   val,
  /* in */ int64_t       num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ALL(arr, rel, val, num, i, _are)

  return _are;
} /* pce_all_longdouble */



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
  int64_t i;

  CONTRACTS_PCE_ALL(arr, "==", NULL, num, i, _are)

  return _are;
}  /* pce_all_null */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_char */




/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The double array variable.
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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_double */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_float */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_int */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_int64 */



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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_long */



/**
 * Determine if any (of the first) num entries have the specified relation
 * to the value.
 *
 * @param[in] arr  The long double array variable.
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
  /* in */ int64_t      num)
{
  CONTRACTS_BOOL _are = CONTRACTS_FALSE;
  int64_t i;

  CONTRACTS_PCE_ANY(arr, rel, val, num, i, _are)

  return _are;
}  /* pce_any_longdouble */



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
  int64_t i;

  CONTRACTS_PCE_ANY(arr, "==", NULL, num, i, _are)

  return _are;
}  /* pce_any_null */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_char */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_double */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_float */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_int */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_int64_t */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_long */



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
  /* in */ int64_t      num)
{
  int64_t _cnt, i;

  CONTRACTS_PCE_COUNT(arr, rel, val, num, i, _cnt)

  return _cnt;
}  /* pce_count_longdouble */



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
 * Determine if the provided variable is within the tolerance of the the 
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
  /* in */ long double tol)
{
  return ((minvalue-tol) <= var) && (var <= (maxvalue+tol));
}  /* pce_range */


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
