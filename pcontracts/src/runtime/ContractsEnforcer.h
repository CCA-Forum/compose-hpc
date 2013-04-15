/**
 * \internal
 * File:           ContractsEnforcer.h
 * Author:         T. Dahlgren
 * Created:        2012 May 11
 * Last Modified:  2013 April 9
 * \endinternal
 *
 * @file
 * @brief 
 * Interface contract enforcement manager.
 *
 * @htmlinclude contractsSource.html
 * @htmlinclude copyright.html
 */

#ifndef ContractsEnforcer_h
#define ContractsEnforcer_h

#include "contracts.h"
#include "contractOptions.h"
#include "contractPrivateTypes.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * \publicsection 
 */

#ifdef PAUL_CONTRACTS
/**
 * Macro to check an assertion expression.
 */
#define PCE_CHECK_EXPR(ENF, TP, TA, TR, LBL, EXPR, CVE) { \
  if (ContractsEnforcer_enforceClause((ENF), (TP), (TA), (TR))) { \
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
#define PCE_CHECK_EXPR_TERM(ENF, TP, TA, TR, LBL, EXPR) { \
  ContractViolationEnum _pce_vio = ContractViolation_NONE; \
  PCE_CHECK_EXPR((ENF), (TP), (TA), (TR), (LBL), (EXPR), _pce_vio) \
  if (_pce_vio != ContractViolation_NONE) { exit(1); } \
}

#define PCE_DUMP_STATS(ENF,CMT) ContractsEnforcer_dumpStatistics(ENF, CMT);
#define PCE_FINALIZE() ContractsEnforcer_finalize();
#define PCE_INITIALIZE(FN) ContractsEnforcer_initialize(FN);
#define PCE_UPDATE_EST_TIME(ENF,TR) ContractsEnforcer_updateEstTime(ENF, TR);

#else /* !def PAUL_CONTRACTS */

#define PCE_CHECK_EXPR(ENF, TP, TA, TR, LBL, EXPR, CVE) 
#define PCE_CHECK_EXPR_TERM(ENF, TP, TA, TR, LBL, EXPR)
#define PCE_DUMP_STATS(ENF,CMT) 
#define PCE_FINALIZE() 
#define PCE_INITIALIZE(FN)
#define PCE_UPDATE_EST_TIME(ENF,TR)
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
  /** Basic enforcement policy */
  EnforcementPolicyType   policy;
  /** Enforcement state */
  EnforcementStateType    data;
  /** [Optional] Enforcement statistics */
  EnforcementFileType*    stats;
  /** [Optional] Enforcement tracing */
  EnforcementFileType*    trace;
} ContractsEnforcerType;


/**
 * Configuration file name. 
 */
extern const char*            pce_config_filename;

/**
 * Contracts enforcer "instance".
 */
extern ContractsEnforcerType* pce_enforcer;

/**
 * Default contracts enforcement-related time estimates.
 */
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
 * @param[in] configfile [Optional] Name of the contract enforcement 
 *                         configuration file.
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
 * @param[in] clauses   Clause(s) to be checked when encountered.
 * @param[in] frequency Frequency of checking encountered clauses.
 * @param[in] value     The policy value option, when appropriate.
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] clauses   Clause(s) to be checked every time they are encountered.
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] clauses   Clause(s) to be checked at the specified interval.
 * @param[in] interval  The desired check frequency (i.e., for each interval
 *                        clause encountered).
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] clauses   Clause(s) to be checked at the specified interval.
 * @param[in] window    The maximum size of the runtime check window.
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] clauses   Clause(s) to be checked at the specified interval.
 * @param[in] limit     Runtime overhead limit, from 1 to 99, as a percentage of
 *                        execution time.  If 0, the value used will default to
 *                        1 or, if greater than 99, to 99.
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] clauses   Clause(s) to be checked at the specified interval.
 * @param[in] limit     Runtime overhead limit, from 1 to 99, as a percentage of
 *                        execution time.  If 0, the value used will default to
 *                        1 or, if greater than 99, to 99.
 * @param[in] statsfile [Optional] Name of the file to output enforcement data.
 * @param[in] tracefile [Optional] Name of the file to output enforcement 
 *                        traces.
 * @return              Pointer to the initialized enforcer, if successful.
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
 * @param[in] enforcer The responsible contracts enforcer.
 * @param[in] msg      [Optional] Message associated with statistics.
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
 * @param enforcer [inout] The responsible contracts enforcer.
 */
void
ContractsEnforcer_free(
  /* inout */ ContractsEnforcerType* enforcer);


/**
 * /privatesectoin
 *
 * FOR INTERNAL/AUTOMATED-USE ONLY.
 *
 * Log a method/routine enforcement estimates into the trace file, if any.
 *
 * @todo How should this work IF decide to actually include it?
 *
 * @param[in] enforcer The responsible contracts enforcer.
 * @param[in] times    Execution time values associated with the routine.
 * @param[in] name     [Optional] Name of the class and/or method/routine whose 
 *                       timing data is to be logged; otherwise, a default is
 *                       provided. [default=TRACE]
 * @param[in] msg      [Optional] Message associated with the trace. 
 *                       [default=""]
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
 * @param     enforcer     [inout] The responsible contracts enforcer.
 * @param[in] clause       The clause whose enforcement is being assessed.
 * @param[in] clauseTime   The time it is estimated to take to check the clause.
 * @param[in] routineTime  The time it is estimated to take to execute the 
 *                           routine body.
 * @return                 CONTRACTS_TRUE if the clause is to be checked; 
 *                           CONTRACTS_FALSE otherwise.
 */
CONTRACTS_BOOL
ContractsEnforcer_enforceClause(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    ContractClauseEnum     clause,
  /* in */    uint64_t               clauseTime,
  /* in */    uint64_t               routineTime);


/**
 * Add the routine time estimate.  This data is needed by partial
 * enforcement strategies.
 *
 * @param enforcer [inout] The responsible contracts enforcer.
 * @param[in] routineTime  The time it is estimated to take to execute the 
 *                           routine body.
 *
 * \warning For internal/automated use \em only.
 */
void
ContractsEnforcer_updateEstTime(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    uint64_t               routineTime);


#ifdef __cplusplus
}
#endif

#endif /* ContractsEnforcer_h */
