/**
 * \internal
 * File:           contracts.h
 * Author:         T. Dahlgren
 * Created:        2012 April 12
 * Last Modified:  2013 March 12
 * \endinternal
 * 
 * @file
 * @brief
 * Core data types and macros for interface contract enforcement.
 *
 * @htmlinclude contractsSource.html
 * @htmlinclude copyright.html
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
  /** NONE:  No violation occurred. */
  ContractViolation_NONE           = 0,
  /** INVARIANT:  An invariant clause was violated. */
  ContractViolation_INVARIANT      = 1,
  /** PRECONDITION:  A precondition clause was violated. */
  ContractViolation_PRECONDITION   = 2,
  /** POSTCONDITION:  A postcondition clause was violated. */
  ContractViolation_POSTCONDITION  = 4,
  /** ASSERT:  An assertion clause was violated. */
  ContractViolation_ASSERT         = 8
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
                                   = ContractViolation_ASSERT;

/**
 * Names corresponding to (and indexable by) ContractViolationEnum.
 *
 * @todo  Consider an alternative...
 */
static const char* S_CONTRACT_VIOLATION[9] = {
  "None",
  "Invariant",
  "Precondition",
  "**undefined1**",
  "Postcondition",
  "**undefined2**",
  "**undefined3**",
  "**undefined4**",
  "Assertion"
};

/** 
 * The minimum Contract Violation name index.  Provided for traversal purposes.
 */
static const unsigned int S_CONTRACT_VIOLATION_MIN_IND = 0;

/** 
 * The maximum Contract Violation name index.  Provided for traversal purposes.
 */
static const unsigned int S_CONTRACT_VIOLATION_MAX_IND = 8;


#ifdef __cplusplus
}
#endif

#endif /* contracts_h */
