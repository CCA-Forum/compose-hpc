/*
 * File:          ContractsEnforcer.h
 * Description:   Interface contract enforcement manager
 * Source:        Based heavily on Babel's sidl_Enforcer and sidl_EnfPolicy.
 *
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the lawrence Livermore National Laboratory.
 * All rights reserved.
 */

#ifndef ContractsEnforcer_h
#define ContractsEnforcer_h

#include "contracts.h"
#include "contractOptions.h"
#include "contractPrivateTypes.h"

#ifdef __cplusplus
 "C" {
#endif

/*
 **********************************************************************
 * ATTRIBUTES/DATA
 **********************************************************************
 */

typedef struct ContractsEnforcer__struct {
  EnforcementPolicyType   policy;    /* Basic enforcement policy */
  EnforcementStateType    data;      /* Enforcement state */
  EnforcementFileType*    stats;     /* [Optional] Enforcement statistics */
  EnforcementFileType*    trace;     /* [Optional] Enforcement tracing */
} ContractsEnforcerType;


/*
 **********************************************************************
 * METHODS/ROUTINES
 **********************************************************************
 */

/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to always check the specified 
 * contract clause(s) and, optionally, output enforcement statistics 
 * and/or tracing data to the specified files.
 * 
 * @param clauses    Clause(s) to be checked every time they are encountered
 * @param statsfile  [Optional] Name of the file to output enforcement data
 * @param tracefile  [Optional] Name of the file to output enforcement traces
 * @return           Pointer to a contracts enforcer used to check contracts
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforceAll(
  /* in */ EnforcementClauseEnum clauses,
  /* in */ const char*           statsfile,
  /* in */ const char*           tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to NEVER check any contract clauses.
 * 
 * @return  Pointer to a contracts enforcer used to check contracts
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforceNone(void);


/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to check the specified contract 
 * clause(s) at the given period and, optionally, output enforcement 
 * statistics and/or tracing data to the specified files.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval
 * @param interval   The desired check frequency (i.e., for each interval
 *                     clause encountered)
 * @param statsfile  [Optional] Name of the file to output enforcement data
 * @param tracefile  [Optional] Name of the file to output enforcement traces
 * @return           Pointer to a contracts enforcer used to check contracts
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforcePeriodic(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           interval,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to check the specified contract 
 * clause(s) at a random location within the specified window and, 
 * optionally, output enforcement statistics and/or tracing data to the 
 * specified files.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval
 * @param window     The maximum size of the runtime check window
 * @param statsfile  [Optional] Name of the file to output enforcement data
 * @param tracefile  [Optional] Name of the file to output enforcement traces
 * @return           Pointer to a contracts enforcer used to check contracts
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforceRandom(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           window,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to check the specified contract 
 * clause(s) ONLY when their estimated execution time does not exceed
 * the given limit of the estimated time spent executing a routine.
 * Enforcement statistics and/or tracing data are optionally output to the 
 * specified files.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval.
 * @param limit      Runtime overhead limit, from 1 to 99, as a percentage of 
 *                     execution time.  If 0, the value used will default to 1
 *                     or, if greater than 99, to 99.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to a contracts enforcer used to check contracts.
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforceAdaptiveFit(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           limit,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Creates an enforcer whose policy is to check the specified contract 
 * clause(s) ONLY when their estimated execution time does not result in 
 * exceeding the given limit of the estimated execution time so far.
 * Enforcement statistics and/or tracing data are optionally output to the 
 * specified files.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval
 * @param limit      Runtime overhead limit, from 1 to 99, as a percentage of 
 *                     execution time.  If 0, the value used will default to 1
 *                     or, if greater than 99, to 99.
 * @param statsfile  [Optional] Name of the file to output enforcement data
 * @param tracefile  [Optional] Name of the file to output enforcement traces
 * @return           Pointer to a contracts enforcer used to check contracts
 */
CONTRACTS_INLINE
ContractsEnforcerType*
ContractsEnforcer_setEnforceAdaptiveTiming(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           limit,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Dumps the enforcement statistics (so far) into the enforcer's statistics
 * file, if any.
 *
 * @param enforcer    The responsible contracts enforcer.
 * @param msg         [Optional] Message associated with statistics.
 */
void
ContractsEnforcer_dumpStatistics(
  /* in */ ContractsEnforcerType* enforcer,
  /* in */ const char*            msg);


/**
 * FOR APPLICATION USE.  
 *
 * Finalizes enabled enforcement features prior to cleaning up and
 * freeing associated memory.
 *
 * @param enforcer    The responsible contracts enforcer.
 */
void
ContractsEnforcer_free(
  /* inout */ ContractsEnforcerType* enforcer);


/**
 * TBD/ToDo:  How should this work IF decide to actually include it?
 *
 * FOR APPLICATION USE.  
 *
 * Logs a method/routine enforcement estimates into the trace file, if any.
 *
 * @param enforcer  The responsible contracts enforcer.
 * @param times     Execution time values associated with the routine.
 * @param name      [Optional] Name of the class and/or method/routine whose 
 *                    timing data is to be logged; otherwise, a default is
 *                    provided. [default=TRACE]
 * @param msg       [Optional] Message associated with the trace. [default=""]
 */
CONTRACTS_INLINE
void
ContractsEnforcer_logTrace(
  /* in */ ContractsEnforcerType* enforcer,
  /* in */ TimeEstimatesType      times,
  /* in */ const char*            name,
  /* in */ const char*            msg);


/**
 * FOR INTERNAL/AUTOMATED-USE ONLY.
 *
 * Returns TRUE if enforcement options are set to check at least a subset
 * of the instrumented calls; otherwise, returns FALSE.
 *
 * @param enforcer     The responsible contracts enforcer.
 * @param clause       The clause whose enforcement is being assessed.
 * @param clauseTime   The time it is estimated to take to check the clause
 * @param routineTime  The time it is estimated to take to execute the routine
 *                       body.
 * @param firstForCall First time a clause is checked for the routine.
 * @return             CONTRACTS_TRUE if the clause is to be checked; 
 *                       CONTRACTS_FALSE otherwise.
 */
CONTRACTS_BOOL
ContractsEnforcer_enforceClause(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    ContractClauseEnum     clause,
  /* in */    uint64_t               clauseTime,
  /* in */    uint64_t               routineTime,
  /* in */    CONTRACTS_BOOL         firstForCall);


#ifdef __cplusplus
}
#endif

#endif /* ContractsEnforcer_h */
