/*
 * File:          ContractsPolicy.h
 * Description:   Interface contract enforcement policy
 * Source:        Borrowed heavily from Babel's sidl_EnfPolicy
 * 
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 */

#ifndef ContractsPolicy_h
#define ContractsPolicy_h

#include "contracts.h"
#include "contractOptions.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 **********************************************************************
 *                      PUBLIC METHODS/ROUTINES                       *
 **********************************************************************
 */

/**
 * Sets the enforcement policy to always check the specified contract clauses.
 * 
 * @param clauses  Enforcement clause(s) [Default = ALL]
 */
void
ContractsPolicy_setEnforceAll(
  /* in */ EnforcementClauseEnum clauses);


/**
 * Sets the enforcement policy to never check any contract clauses.
 */
void
ContractsPolicy_setEnforceNone(void);


/**
 * Sets the enforcement policy to periodically check the specified
 * contract clauses for any instrumented method encountered during
 * runtime using the specified interval.
 * 
 * @param clauses    Enforcement clause(s) [Default = ALL]
 * @param interval   The desired check frequency
 */
void
ContractsPolicy_setEnforcePeriod(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           interval);


/**
 * Sets the enforcement policy to a simple random check, within the specified
 * window, the contract clauses of any instrumented method encountered during
 * runtime.
 * 
 * @param clauses     Enforcement clause(s) [Default = ALL]
 * @param windowSize  The size of the "window" in which to pick a random number
 */
void
ContractsPolicy_setEnforceRandom(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           windowSize);


/**
 * Sets the enforcement policy to an adaptive approach limiting the runtime
 * overhead based on estimated execution times for contract checks versus 
 * time spent in instrumented methods.
 * 
 * @param clauses        Enforcement clause(s) [Default = ALL]
 * @param overheadLimit  Runtime overhead limit [0.0 .. 1.0)
 */
void
ContractsPolicy_setEnforceAdaptive(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ double                 overheadLimit);


/**
 * Returns TRUE if enforcement options are set to check at least a subset
 * of the instrumented calls; otherwise, returns FALSE.
 */
CONTRACTS_BOOL
ContractsPolicy_areEnforcing(void);


/**
 * Returns the enforcement clause(s) associated with the current policy
 * settings.
 */
EnforcementClauseEnum
ContractsPolicy_getClauses(void);


/**
 * Returns the period, if any, associated with the current policy settings.
 */
unsigned int
ContractsPolicy_getPeriod(void);


/**
 * Returns the random window, if any, associated with the current policy 
 * settings.
 */
unsigned int
ContractsPolicy_getRandomWindow(void);


/**
 * Returns the overhead limit, if any, associated with the current policy
 * settings.
 */
double
ContractsPolicy_getOverheadLimit(void);


/**
 * Returns the policy name associated with the current policy settings.
 */
char*
ContractsPolicy_getPolicyName(void);


#if 0
/* TBD/TODO:  Is this capability going to be retained? */

/**
 * Prints statistics data to the file with the specified name.
 * The file is opened (for append) and closed on each call.
 *
 * @param filename    Name of the statistics output file.
 * @param compressed  TRUE for semi-colon separated output.
 */
void
ContractsPolicy_dumpStats(
  /* in */ const char*     filename,
  /* in */ CONTRACTS_BOOL  compressed);


/**
 * Starts enforcement trace file generation.
 * 
 * @param filename   Name of the destination trace file.
 * @param trace      Desired contract enforcement tracing level [Default=NONE]
 */
void
ContractsPolicy_startTrace(
  /* in */ const char*        filename,
  /* in */ ContractTraceEnum  trace);


/**
 * Returns TRUE if contract enforcement tracing is enabled;
 * FALSE otherwise.
 */
CONTRACTS_BOOL
ContractsPolicy_areTracing(void);


/**
 * Returns the name of the trace file.  If one was not provided,
 * the default name is returned.
 */
char*
ContractsPolicy_getTraceFilename(void);


/**
 * Returns the level of enforcement tracing.
 */
ContractTraceEnum
ContractsPolicy_getTraceLevel(void);

/**
 * Terminates enforcement tracing.  Takes a final timestamp and logs the 
 * remaining trace information.
 */
void
ContractsPolicy_endTrace(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ContractsPolicy_h */
