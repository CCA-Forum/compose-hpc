/*
 * File:           contractMacros.h
 * Author:         T. Dahlgren
 * Created:        2012 June 8
 * Last Modified:  2012 November 12
 *
 *
 * @section DESCRIPTION
 * Convenience C macros for managing contract enforcement.
 *
 * @todo Is this file even relevant in this context?
 *
 *
 * @section SOURCE
 * This file is based heavily on Babel's sidlAsserts.h.
 *
 *
 * @section REQUIREMENTS
 * The following include files are needed:
 *    math.h      For the ceiling function used by the 
 *                  random and timing-based policies.
 *    stdlib.h    For random number generation (including RAND_MAX).
 *    time.h      For processing associated with the timing-based policy.
 *
 *
 * @section COPYRIGHT
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Tamara Dahlgren <dahlgren1@llnl.gov>.
 * 
 * LLNL-CODE-473891.
 * All rights reserved.
 * 
 * This software is part of COMPOSE-HPC. See http://compose-hpc.sourceforge.net/
 * for details.  Please read the COPYRIGHT file for Our Notice and for the 
 * BSD License.
 */

#ifndef contractMacros_h
#define contractMacros_h

#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

/****************************************************************************
 * Contract assertion checking macros for interface contract enforcement.
 ****************************************************************************/

/*
 * CONTRACT_ARRAY_ALL_BOTH   all(a1 r a2), where a1 and a2 are arrays, r is 
 *                           the relation
 */
#define CONTRACT_ARRAY_ALL_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C, BRES) \
   CONTRACT_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_ALL_VL   all(vr a), where a is array, vr is value + relation
 */
#define CONTRACT_ARRAY_ALL_VL(AC, AV, REL, I, NUM, C, BRES) {\
   CONTRACT_ARRAY_COUNT_VL(AC, AV, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_ALL_VR   all(a rv), where a is array, rv is relation + value
 */
#define CONTRACT_ARRAY_ALL_VR(AC, AV, REL, I, NUM, C, BRES) {\
   CONTRACT_ARRAY_COUNT_VR(AC, AV, REL, I, NUM, C);\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_ANY_BOTH   any(a1 r a2), where a1 and a2 are arrays, r is the
 *                           relation
 *
 *   NOTE: Will return FALSE if the arrays are not the same size.
 */
#define CONTRACT_ARRAY_ANY_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = CONTRACT_ARRAY_SIZE(AC1, (AV1));\
   if (CONTRACT_ARRAY_SIZE(AC2, (AV2)) == NUM) {\
     for (I=0; (I<NUM) && (!BRES); I++) {\
       CONTRACT_INCR_IF_TRUE((AC1##_get1((AV1),I) REL AC2##_get1((AV2),I)), BRES)\
     }\
   }\
}

/*
 * CONTRACT_ARRAY_ANY_VL   any(vr a), where a is array, vr is value + relation
 */
#define CONTRACT_ARRAY_ANY_VL(AC, AV, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (!BRES); I++) {\
     CONTRACT_INCR_IF_TRUE((REL AC##_get1((AV),I)), BRES)\
   }\
}

/*
 * CONTRACT_ARRAY_ANY_VR   any(a rv), where a is array, rv is relation + value
 */
#define CONTRACT_ARRAY_ANY_VR(AC, AV, REL, I, NUM, BRES) {\
   BRES = FALSE;\
   NUM  = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (!BRES); I++) {\
     CONTRACT_INCR_IF_TRUE((AC##_get1((AV),I) REL), BRES)\
   }\
}

/*
 * CONTRACT_ARRAY_COUNT_BOTH  count(a1 r a2), where a1 and a2 are arrays, r is 
 *                            the relation.
 *
 *   NOTE: Will return FALSE if the arrays are not the same size.
 */
#define CONTRACT_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC1, (AV1));\
   if (CONTRACT_ARRAY_SIZE(AC2, (AV2)) == NUM) {\
     for (I=0; I<NUM; I++) {\
       CONTRACT_INCR_IF_TRUE((AC1##_get1((AV1),I) REL AC2##_get1((AV2),I)), IRES)\
     }\
   }\
}

/*
 * CONTRACT_ARRAY_COUNT_VL   count(vr a), where a is array, vr is value + 
 *                           relation
 */
#define CONTRACT_ARRAY_COUNT_VL(AC, AV, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     CONTRACT_INCR_IF_TRUE((REL AC##_get1((AV),I)), IRES)\
   }\
}

/*
 * CONTRACT_ARRAY_COUNT_VR   count(a rv), where a is array, rv is relation + 
 *                           value
 */
#define CONTRACT_ARRAY_COUNT_VR(AC, AV, REL, I, NUM, IRES) {\
   IRES = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     CONTRACT_INCR_IF_TRUE((AC##_get1((AV),I) REL), IRES)\
   }\
}

/*
 * CONTRACT_ARRAY_DIMEN	   dimen(a), where a is the array
 */
#define CONTRACT_ARRAY_DIMEN(AC, AV) sidlArrayDim(AV)

/*
 * CONTRACT_ARRAY_IRANGE   irange(a, v1, v2), where a is array whose integer
 *                        values are to be in v1..v2.
 */
#define CONTRACT_ARRAY_IRANGE(AC, AV, V1, V2, I, NUM, C, BRES) {\
   C   = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     CONTRACT_INCR_IF_TRUE(\
       CONTRACT_IRANGE((double)AC##_get1((AV),I), (double)V1, (double)V2), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_LOWER   lower(a, d), where a is the array and d is the 
 *                        dimension
 */
#define CONTRACT_ARRAY_LOWER(AC, AV, D) AC##_lower((AV), D)

/*
 * CONTRACT_ARRAY_MAX   max(a), where a is the array of scalar
 */
#define CONTRACT_ARRAY_MAX(AC, AV, I, NUM, T, RES) {\
   RES  = AC##_get1((AV),0);\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) {\
     T _SAMAXV = AC##_get1((AV),I);\
     if (_SAMAXV > RES) { RES = _SAMAXV; } \
   }\
}

/*
 * CONTRACT_ARRAY_MIN   min(a), where a is the array of scalar
 */
#define CONTRACT_ARRAY_MIN(AC, AV, I, NUM, T, RES) {\
   RES  = AC##_get1((AV),0);\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) {\
     T _SAMINV = AC##_get1((AV),I);\
     if (_SAMINV < RES) { RES = _SAMINV; } \
   }\
}

/*
 * CONTRACT_ARRAY_NEAR_EQUAL   nearEqual(a, b, tol), where a and b are arrays
 *                             whose scalar values are to be compared.
 */
#define CONTRACT_ARRAY_NEAR_EQUAL(AC1, AV1, AC2, AV2, TOL, I, NUM, C, BRES) {\
   C = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC1, (AV1));\
   for (I=0; I<NUM; I++) {\
     CONTRACT_INCR_IF_TRUE(\
       CONTRACT_NEAR_EQUAL(AC1##_get1((AV1),I), AC2##_get1((AV2),I), TOL), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_NON_INCR   nonIncr(a), where a is array of numeric values
 *                           to be checked for being in decreasing order.
 */
#define CONTRACT_ARRAY_NON_INCR(AC, AV, I, NUM, V, BRES) {\
   BRES = TRUE;\
   V    = ((AV) != NULL) ? (double) AC##_get1((AV),0) : 0.0;\
   NUM  = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (BRES); I++) {\
     if ((double)AC##_get1((AV),I) > V) {\
       BRES = FALSE; \
     } else {\
       V = (double) AC##_get1((AV),0);\
     }\
   }\
}

/*
 * CONTRACT_ARRAY_NONE_BOTH   none(a1 r a2), where a1 and a2 are arrays, r is
 *                            the relation.
 */
#define CONTRACT_ARRAY_NONE_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C, BRES) {\
   CONTRACT_ARRAY_COUNT_BOTH(AC1, AV1, AC2, AV2, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * CONTRACT_ARRAY_NONE_VL   none(vr a), where a is array, vr is value + relation
 */
#define CONTRACT_ARRAY_NONE_VL(AC, AV, REL, I, NUM, C, BRES) {\
   CONTRACT_ARRAY_COUNT_VL(AC, AV, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * CONTRACT_ARRAY_NONE_VR   none(a rv), where a is array, rv is relation + value
 */
#define CONTRACT_ARRAY_NONE_VR(AC, AV, REL, I, NUM, C, BRES) {\
   CONTRACT_ARRAY_COUNT_VR(AC, AV, REL, I, NUM, C);\
   BRES = (C == 0);\
}

/*
 * CONTRACT_ARRAY_RANGE   range(a, v1, v2, tol), where a is array whose scalar
 *                        values are to be in v1..v2 within tolerance tol.
 */
#define CONTRACT_ARRAY_RANGE(AC, AV, V1, V2, TOL, I, NUM, C, BRES) {\
   C   = 0;\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; I<NUM; I++) {\
     CONTRACT_INCR_IF_TRUE(\
       CONTRACT_RANGE((double)AC##_get1((AV),I), (double)V1, (double)V2, TOL), C)\
   }\
   BRES = (C == NUM);\
}

/*
 * CONTRACT_ARRAY_SIZE   size(a), where a is the array 
 */
#define CONTRACT_ARRAY_SIZE(AC, AV) sidlLength(AV, 0)

/*
 * CONTRACT_ARRAY_STRIDE   stride(a, d), where a is the array and d is the 
 *                         dimension
 */
#define CONTRACT_ARRAY_STRIDE(AC, AV, D) sidlStride(AV, D)

/*
 * CONTRACT_ARRAY_SUM   sum(a), where a is the array of scalar
 */
#define CONTRACT_ARRAY_SUM(AC, AV, I, NUM, RES) {\
   RES = AC##_get1((AV),0);\
   NUM = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=1; I<NUM; I++) { RES += AC##_get1((AV),I); }\
}

/*
 * CONTRACT_ARRAY_NON_DECR   nonDecr(a), where a is array of numeric values
 *                           to be checked for being in increasing order.
 */
#define CONTRACT_ARRAY_NON_DECR(AC, AV, I, NUM, V, BRES) {\
   BRES = TRUE;\
   V    = ((AV) != NULL) ? (double) AC##_get1((AV), 0) : 0.0;\
   NUM  = CONTRACT_ARRAY_SIZE(AC, (AV));\
   for (I=0; (I<NUM) && (BRES); I++) {\
     if ((double)AC##_get1((AV),I) < V) {\
       BRES = FALSE; \
     } else {\
       V = (double) AC##_get1((AV), 0);\
     }\
   }\
}

/*
 * CONTRACT_ARRAY_UPPER   upper(a, d), where a is the array and d is the 
 *                        dimension
 */
#define CONTRACT_ARRAY_UPPER(AC, AV, D) sidlUpper(AV, D)

/*
 * CONTRACT_IRANGE   irange(v, v1, v2), where determine if v in the 
 *                   range v1..v2.
 */
#define CONTRACT_IRANGE(V, V1, V2) \
   (  ((double)V1 <= (double)V) && ((double)V  <= (double)V2) ) 

/*
 * CONTRACT_NEAR_EQUAL   nearEqual(v1, v2, tol), where v1 and v2 are scalars 
 *                       being checked for being equal within the specified 
 *                       tolerance, tol.
 */
#define CONTRACT_NEAR_EQUAL(V1, V2, TOL)  \
   (fabs((double)V1 - (double)V2) <= (double)TOL)

/*
 * CONTRACT_RANGE   range(v, v1, v2, tol), where determine if v in
 *                  the range v1..v2, within the specified tolerance, tol.
 */
#define CONTRACT_RANGE(V, V1, V2, TOL) {\
   (  (((double)V1 - (double)TOL) <= (double)V) \
   && ((double)V                  <= ((double)V2 + (double)TOL)) ) \
}


/****************************************************************************
 * Additional macros
 ****************************************************************************/

/*
 *  CONTRACT_DIFF_MICROSECONDS   "Standard" time difference
 */
#define CONTRACT_DIFF_MICROSECONDS(T2, T1) \
  (1.0e6*(double)(T2.tv_sec-T1.tv_sec)) + (T2.tv_usec-T1.tv_usec)

/*
 *  CONTRACT_INCR_IF_THEN   Increment V1 if EXPR is TRUE; otherwise,
 *                          increment V2.
 */
#define CONTRACT_INCR_IF_THEN(EXPR, V1, V2) \
  if (EXPR) { (V1) += 1; } else { (V2) += 1; }

/*
 *  CONTRACT_INCR_IF_TRUE   Increment V if EXPR is TRUE.
 */
#define CONTRACT_INCR_IF_TRUE(EXPR, V)  if (EXPR) { (V) += 1; } 

#endif /* contractMacros_h */
