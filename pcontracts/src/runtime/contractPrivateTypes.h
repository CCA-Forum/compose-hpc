/**
 * \internal
 * File:           contractPrivateTypes.h
 * Author:         T. Dahlgren
 * Created:        2012 April 23
 * Last Modified:  2012 November 28
 * \endinternal
 * 
 * @file
 * @brief 
 * "Private" types associated with interface contract enforcement. 
 *
 * @htmlinclude contractsSource.html
 * @htmlinclude copyright.html
 */

#ifndef contractPrivateTypes_h
#define contractPrivateTypes_h

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include "contractOptions.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Interface contract enforcement policy data.
 */
typedef struct EnforcementPolicy__struct {
  /** Types of clauses to be checked. */
  EnforcementClauseEnum       clauses;   
  /** Frequency of clause checking. */
  EnforcementFrequencyEnum    frequency; 
  /** Policy-based checking value. */
  unsigned int                value;     
} EnforcementPolicyType;


/**
 * Interface contract enforcement time estimates, in milliseconds.  This 
 * data is ONLY applicable to adaptive enforcement policy(ies).
 */
typedef struct TimeEstimates__struct {
  /** Milliseconds spent checking precondition clause. */
  uint64_t pre;     
  /** Milliseconds spent checking postcondition clause. */
  uint64_t post;    
  /** Milliseconds spent checking invariant clause. */
  uint64_t inv;     
  /** Milliseconds spent in the routine implementation. */
  uint64_t routine; 
} TimeEstimatesType;


/**
 * Interface contract enforcement (management) state.  
 */
typedef struct EnforcementState__struct {
  /** Number of enforcement requests. */
  uint64_t           requests;   
  /** Number of allowed enforcement requests. */
  uint64_t           allowed;    
  /** Total checks time (derivable). */
  uint64_t           checksTime; 
  /** Countdown used for basic sampling. */
  unsigned int       countdown;  
  /** Skip, if any, in a random window. */
  unsigned int       skip;       
  /** TBD?  Starting timestamp. */
  struct timeval     start;      
  /** Timing accumulators. */
  TimeEstimatesType  total;      
} EnforcementStateType;


/**
 * Basic interface contract enforcement file data.  
 */
typedef struct EnforcementFile__struct {
  /** Name of the file. */
  char*    fileName;  
  /** Pointer to the file. */
  FILE*    filePtr;   
} EnforcementFileType;


#ifdef __cplusplus
}
#endif

#endif /* contractPrivateTypes_h */
