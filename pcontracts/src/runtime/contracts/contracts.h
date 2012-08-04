/**
 * File:  contracts.h
 * 
 * @file
 * @section DESCRIPTION
 * Interface contract enforcement basis, which includes core macros and types.
 *
 * @section LICENSE
 * TBD
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

/**
 * Contract enforcement (clause) violation types.
 *
 * WARNING: Expected to be kept in sync with corresponding 
 *   ContractClause__enum values.
 */
typedef enum ContractViolation__enum {
  /** No violation occurred. */
  ContractViolation_NONE           = 0,
  /** An invariant clause was violated. */
  ContractViolation_INVARIANT      = 1,
  /** A precondition clause was violated. */
  ContractViolation_PRECONDITION   = 2,
  /** A postcondition clause was violated. */
  ContractViolation_POSTCONDITION  = 4,
  /** Future Work Placeholder */
  ContractViolation_CUSTOM         = 8
} ContractViolationEnum;

/** 
 * The minimum Contract Violation enumeration number.  Provided
 * for traversal purposes.
 */
static const ContractViolationEnum S_CONTRACT_VIOLATION_MIN
                                   = ContractViolation_NONE;

/** 
 * The maximum Contract Violation enumeration number.  Provided
 * for traversal purposes.
 */
static const ContractViolationEnum S_CONTRACT_VIOLATION_MAX
                                   = ContractViolation_CUSTOM;

/**
 * Names corresponding to (and indexable by) ContractViolationEnum.
 *
 * @todo  Consider an alternative...
 */
static const char* S_CONTRACT_VIOLATION[5] = {
  "None",
  "Invariant",
  "Precondition",
  "**undefined**",
  "Postcondition",
  "**undefined**",
  "**undefined**",
  "**undefined**",
  "Custom"
};

/** 
 * The minimum Contract Violation name index.  Provided for traversal purposes.
 */
static const unsigned int S_CONTRACT_VIOLATION_MIN_IND = 0;

/** 
 * The maximum Contract Violation name index.  Provided for traversal purposes.
 */
static const unsigned int S_CONTRACT_VIOLATION_MAX_IND = 4;


#ifdef __cplusplus
}
#endif

#endif /* contracts_h */
