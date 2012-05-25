/*
 * File:          contractPrivateTypes.h
 * Description:   "Private" types associated with interface contract
 *                enforcement. 
 * 
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
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
 * ----------------------------------------------------------------------
 * Interface contract enforcement policy data.
 * ----------------------------------------------------------------------
 */
typedef struct EnforcementPolicy__struct {
  EnforcementClauseEnum       clauses;   /* Types of clauses to be checked */
  EnforcementFrequencyEnum    frequency; /* Frequency of clause checking */
  unsigned int                value;     /* Policy-based checking value */
} EnforcementPolicyType;


/**
 * ----------------------------------------------------------------------
 * Interface contract enforcement (management) state.  
 * ----------------------------------------------------------------------
 */
typedef struct EnforcementState__struct {
  uint64_t           requests;   /* Number of enforcement requests */
  uint64_t           allowed;    /* Number of allowed enforcement requests */
  uint64_t           checksTime; /* Total checks time (derivable) */
  unsigned int       countdown;  /* Countdown used for basic sampling */
  unsigned int       skip;       /* Skip, if any, in a random window */
  struct timeval     start;      /* TBD?  Starting timestamp */
  TimeEstimatesType  total;      /* Timing accumulators */
} EnforcementStateType;


/**
 * ----------------------------------------------------------------------
 * Interface contract enforcement time estimates, in milliseconds.  This 
 * data is ONLY applicable to adaptive enforcement policy(ies).
 * ----------------------------------------------------------------------
 */
typedef struct TimeEstimates__struct {
  uint64_t pre;     /* Milliseconds spent checking precondition clause */
  uint64_t post;    /* Milliseconds spent checking postcondition clause */
  uint64_t inv;     /* Milliseconds spent checking invariant clause */
  uint64_t routine; /* Milliseconds spent in the routine implementation */
} TimeEstimatesType;


/**
 * ----------------------------------------------------------------------
 * Basic interface contract enforcement file data.  
 * ----------------------------------------------------------------------
 */
typedef struct EnforcementFile__struct {
  char*    fileName;  /* Name of the file */
  FILE*    filePtr;   /* Pointer to the file */
} EnforcementFileType;


#ifdef __cplusplus
}
#endif

#endif /* contractPrivateTypes_h */
