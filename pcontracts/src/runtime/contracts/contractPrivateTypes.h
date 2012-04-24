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
  unsigned int                limit;     /* Limit, if any, on checking */
} EnforcementPolicyType;


/**
 * ----------------------------------------------------------------------
 * Interface contract enforcement (management) state.  
 * ----------------------------------------------------------------------
 */
typedef struct EnforcementState__struct {
  uint64_t           requests;  /* Number of enforcement requests */
  uint64_t           allowed;   /* Number of allowed enforcement requests */
  unsigned int       countdown; /* countdown used for basic sampling */
  unsigned int       skip;      /* skip, if any, in a random window */
  TimeEstimatesType  total;     /* Timing accumulators */
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
  uint64_t caller;  /* Milliseconds spent outside annotated routines */
} TimeEstimatesType;


/**
 * ----------------------------------------------------------------------
 * Interface contract enforcement tracing data.  This data is associated
 * with enforcement tracing features.
 * ----------------------------------------------------------------------
 */
typedef struct EnforcementTracing__struct {
  char*    traceFile;  /* Name of the trace file */
  FILE*    tracePtr;   /* Pointer to the trace file */
} EnforcementTracingType;


#ifdef __cplusplus
}
#endif

#endif /* contractPrivateTypes_h */
