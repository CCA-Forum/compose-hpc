/*
 * File:          contracts.h
 * Description:   Interface contract enforcement basis, which includes
 *                core macros and types.
 * 
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 */

#ifndef contracts_h
#define contracts_h

/*
 **********************************************************************
 *                           BASIC MACROS                             *
 **********************************************************************
 */

#ifndef CONTRACTS_BOOL
#if defined(__cplusplus)
#define CONTRACTS_BOOL bool
#ifndef CONTRACTS_TRUE
#define CONTRACTS_TRUE true
#endif
#ifndef CONTRACTS_FALSE
#define CONTRACTS_FALSE false
#endif
#else
#define CONTRACTS_BOOL int
#ifndef CONTRACTS_TRUE
#define CONTRACTS_TRUE 1
#endif
#ifndef CONTRACTS_FALSE
#define CONTRACTS_FALSE 0
#endif
#endif
#endif /* CONTRACTS_BOOL */

#ifndef CONTRACTS_INLINE
#if defined(__cplusplus)
#define CONTRACTS_INLINE inline
#else 
#define CONTRACTS_INLINE 
#endif
#endif /* CONTRACTS_INLINE */

#ifndef NULL
#define NULL 0
#endif

/* TBD/ToDo:  Obsolete? */
#ifndef DIFFT
#define DIFFT(T2, T1) \
  1.0e6*(double)((T2).tv_sec - (T1).tv_sec) \
  + ((T2).tv_usec-(T1).tv_usec)
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*
 **********************************************************************
 *                           BASIC TYPES                              *
 **********************************************************************
 */

/* 
 * ------------------------------------------------------------------
 * Contract enforcement violation types.
 * ------------------------------------------------------------------
 */
typedef enum ContractViolation__enum {
  ContractViolation_NONE           = 0,
  ContractViolation_INVARIANT      = 1,
  ContractViolation_PRECONDITION   = 2,
  ContractViolation_POSTCONDITION  = 3
} ContractViolationEnum;

/*
 * Names corresponding to (and indexed by) ContractViolationEnum.
 */
static const char* S_CONTRACT_VIOLATION[4] = {
  "None",
  "Invariant",
  "Precondition",
  "Postcondition"
};
static const unsigned int S_CONTRACT_VIOLATION_MIN_IND = 0;
static const unsigned int S_CONTRACT_VIOLATION_MAX_IND = 3;


#ifdef __cplusplus
}
#endif

#endif /* contracts_h */
