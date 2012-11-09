/**
 * File:           ContractsEnforcer.h
 * Author:         T. Dahlgren
 * Created:        2012 May 11
 * Last Modified:  2012 October 9
 *
 * @file
 * @section DESCRIPTION
 * Interface contract enforcement manager.
 *
 * @section SOURCE
 * The implementation is based heavily on Babel's sidl_Enforcer and 
 * sidl_EnfPolicy.
 *
 * @section LICENSE
 * TBD
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
extern "C" {
#endif


#ifdef PAUL_CONTRACTS
/**
 * Macro to check an assertion expression.
 */
#define PCE_CHECK_EXPR(ENF, TP, TA, TR, FT, LBL, EXPR, CVE) { \
  if (ContractsEnforcer_enforceClause((ENF), (TP), (TA), (TR), (FT))) { \
    if (!(EXPR)) { \
      printf("ERROR: %s Violation: %s: %d\n", S_CONTRACT_CLAUSE[TP], \
             (LBL), (EXPR)); \
      (CVE) = (ContractViolationEnum)(TP); \
    } \
  } \
}

/**
 * Macro to check an assertion expression and terminate if violated.
 */
#define PCE_CHECK_EXPR_TERM(ENF, TP, TA, TR, FT, LBL, EXPR) { \
  ContractViolationEnum _pce_vio = ContractViolation_NONE; \
  PCE_CHECK_EXPR((ENF), (TP), (TA), (TR), (FT), (LBL), (EXPR), _pce_vio) \
  if (_pce_vio != ContractViolation_NONE) { exit(1); } \
}

#define PCE_DUMP_STATS(ENF,CMT) ContractsEnforcer_dumpStatistics(ENF, CMT);
#define PCE_FINALIZE() ContractsEnforcer_finalize();
#define PCE_INITIALIZE(FN) ContractsEnforcer_initialize(FN);

#else /* !def PAUL_CONTRACTS */

#define PCE_CHECK_EXPR(ENF, TP, TA, TR, FT, LBL, EXPR, CVE) 
#define PCE_CHECK_EXPR_TERM(ENF, TP, TA, TR, FT, LBL, EXPR)
#define PCE_DUMP_STATS(ENF,CMT) 
#define PCE_FINALIZE() 
#define PCE_INITIALIZE(FN)
#endif /* PAUL_CONTRACTS */


/*
 **********************************************************************
 * ATTRIBUTES/DATA
 **********************************************************************
 */

/**
 * Contracts Enforcer data.
 */
typedef struct ContractsEnforcer__struct {
  EnforcementPolicyType   policy;    /** Basic enforcement policy */
  EnforcementStateType    data;      /** Enforcement state */
  EnforcementFileType*    stats;     /** [Optional] Enforcement statistics */
  EnforcementFileType*    trace;     /** [Optional] Enforcement tracing */
} ContractsEnforcerType;


/**
 * Active "instance" data.
 */
extern const char*            pce_config_filename;
extern ContractsEnforcerType* pce_enforcer;
extern TimeEstimatesType      pce_def_times;


/*
 **********************************************************************
 * PUBLIC METHODS/ROUTINES
 **********************************************************************
 */

/**
 * FOR APPLICATION/AUTOMATED INSTRUMENTATION USE.  
 *
 * Create a global enforcer configured based on the optional input file.
 * If no input file is provided, then default contract enforcement options
 * are used.
 *
 * @param configfile [Optional] Name of the contract enforcement configuration
 *                     file.
 */
void
ContractsEnforcer_initialize(
  /* in */ const char* configfile);


/**
 * FOR APPLICATION/AUTOMATED INSTRUMENTATION USE.  
 *
 * Finalize the global enforcer, releasing memory and performing associated
 * clean up.
 */
void
ContractsEnforcer_finalize(void);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for checking the specified contract clause(s) at 
 * the given frequency (and associated value).  Enforcement statistics
 * and/or tracing data are output to the given files, when provided.
 *
 * @param clauses    Clause(s) to be checked when encountered.
 * @param frequency  Frequency of checking encountered clauses.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_createEnforcer(
  /* in */ EnforcementClauseEnum    clauses, 
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for checking all of the specified contract clause(s) 
 * encountered.  Enforcement statistics and/or tracing data are output to 
 * the given files, when provided.
 * 
 * @param clauses    Clause(s) to be checked every time they are encountered.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceAll(
  /* in */ EnforcementClauseEnum clauses,
  /* in */ const char*           statsfile,
  /* in */ const char*           tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer whose policy is to NEVER check any contract clauses.
 * 
 * @return  Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceNone(void);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for periodically checking contract clause(s) at
 * the specified interval.  Enforcement statistics and/or tracing data 
 * are output to the given files, when provided.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval.
 * @param interval   The desired check frequency (i.e., for each interval
 *                     clause encountered).
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforcePeriodic(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           interval,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for randomly the specified checking contract clause(s), 
 * once within each window.  Enforcement statistics and/or tracing data are 
 * output to the given files, when provided.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval.
 * @param window     The maximum size of the runtime check window.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceRandom(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           window,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for adaptively checking contract clause(s) whose 
 * estimated execution times does not exceed the specified percentage limit
 * on the estimated time spent executing a routine.  Enforcement statistics 
 * and/or tracing data are optionally output to the given files.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval.
 * @param limit      Runtime overhead limit, from 1 to 99, as a percentage of 
 *                     execution time.  If 0, the value used will default to 1
 *                     or, if greater than 99, to 99.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceAdaptiveFit(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           limit,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer for adaptively checking the specified contract
 * clause(s) ONLY when their estimated execution time does not result in 
 * exceeding the given limit of the estimated total execution time so far.
 * Enforcement statistics and/or tracing data are output to the given
 * files, when provided.
 * 
 * @param clauses    Clause(s) to be checked at the specified interval.
 * @param limit      Runtime overhead limit, from 1 to 99, as a percentage of 
 *                     execution time.  If 0, the value used will default to 1
 *                     or, if greater than 99, to 99.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceAdaptiveTiming(
  /* in */ EnforcementClauseEnum  clauses,
  /* in */ unsigned int           limit,
  /* in */ const char*            statsfile,
  /* in */ const char*            tracefile);


/**
 * FOR APPLICATION USE.  
 *
 * Dump enforcement statistics (so far) into the enforcer's statistics
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
 * Finalize enabled enforcement features prior to cleaning up and freeing
 * associated memory.
 *
 * @param enforcer    The responsible contracts enforcer.
 */
void
ContractsEnforcer_free(
  /* inout */ ContractsEnforcerType* enforcer);


/**
 * TBD/ToDo:  How should this work IF decide to actually include it?
 *
 * FOR INTERNAL/AUTOMATED-USE ONLY.
 *
 * Log a method/routine enforcement estimates into the trace file, if any.
 *
 * @param enforcer  The responsible contracts enforcer.
 * @param times     Execution time values associated with the routine.
 * @param name      [Optional] Name of the class and/or method/routine whose 
 *                    timing data is to be logged; otherwise, a default is
 *                    provided. [default=TRACE]
 * @param msg       [Optional] Message associated with the trace. [default=""]
 */
void
ContractsEnforcer_logTrace(
  /* in */ ContractsEnforcerType* enforcer,
  /* in */ TimeEstimatesType      times,
  /* in */ const char*            name,
  /* in */ const char*            msg);


/**
 * FOR INTERNAL/AUTOMATED-USE ONLY.
 *
 * Determine if it is time to check contract clause.
 *
 * @param enforcer     The responsible contracts enforcer.
 * @param clause       The clause whose enforcement is being assessed.
 * @param clauseTime   The time it is estimated to take to check the clause.
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
