/**
 * \internal
 * File:  testContractsEnforcer.c
 * \endinternal
 *
 * @file
 * @brief
 * Test suite for ContractsEnforcer.
 *
 * @htmlinclude copyright.html
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ContractsEnforcer.h"


/**
 * Test file types.
 */
typedef enum FileType__enum {
  /** Trace file. */
  FileType_TRACE = 0,
  /** Statistics file. */
  FileType_STATISTICS = 1
} FileTypeEnum;

/**
 * Names corresponding to (and indexable by) FileTypeEnum.
 */
static const char* S_FILE_TYPE[2] = {
  "trace",
  "stats"
};



/**
 * Creates a filename tailored to the provided options.
 *
 * @param[in] clauses    Clause(s) to be checked.
 * @param[in] frequency  Frequency of checking.
 * @param[in] fileType   Type of file being created.
 * @param[in] ext        Desired file extension [Default=csv].
 * @return           The resulting filename.
 */ 
char*
getFilename(
  /* in */ EnforcementClauseEnum    clauses,
  /* in */ EnforcementFrequencyEnum frequency,
  /* in */ FileTypeEnum             fileType,
  /* in */ const char*              ext) 
{
  char* fn = NULL;
  const char* pre = "tce";
  const char* clauseStr = S_ENFORCEMENT_CLAUSE_ABBREV[clauses];
  const char* freqStr = S_ENFORCEMENT_FREQUENCY[frequency];
  const char* typeStr = S_FILE_TYPE[fileType];
  const char* extStr = ((ext != NULL) && (strlen(ext) > 0)) ? ext : "csv";

  int len = strlen(pre) + strlen(clauseStr) + strlen(freqStr)
          + strlen(typeStr) + strlen(extStr) + 5;
  fn = (char*)malloc(len*sizeof(char));
  if (fn != NULL) {
    sprintf(fn, "%s-%s-%s-%s.%s", pre, clauseStr, freqStr, typeStr, extStr);
  } else {
    printf("\nWARNING:  Unable to allocate space to the filename.\n");
  }
  return fn;
} /* getFilename */


/**
 * Mimics contract enforcement calls for a single routine.
 *
 * @param     enforcer [inout] The responsible contracts enforcer.
 * @param[in] clauses  Clause(s) to be checked.
 * @param[in] times    Times associated with the clause and routine.
 * @return             Number of enforced clauses.
 */ 
unsigned int
checkRoutineClauses(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    EnforcementClauseEnum  clauses,
  /* in */    TimeEstimatesType      times)
{
  unsigned int   numEnforced = 0;

  ContractsEnforcer_updateEstTime(enforcer, times.routine);

  if (clauses & EnforcementClause_INVARIANTS)
  {
    numEnforced += 
      ContractsEnforcer_enforceClause(enforcer, ContractClause_INVARIANT,
        times.inv, times.routine) ? 1 : 0;
  }

  if (clauses & EnforcementClause_PRECONDITIONS)
  {
    numEnforced += 
      ContractsEnforcer_enforceClause(enforcer, ContractClause_PRECONDITION,
        times.pre, times.routine) ? 1 : 0;
  }

  if (clauses & EnforcementClause_POSTCONDITIONS)
  {
    numEnforced += 
      ContractsEnforcer_enforceClause(enforcer, ContractClause_POSTCONDITION,
        times.post, times.routine) ?  1 : 0;
  }

  if (clauses & EnforcementClause_INVARIANTS)
  {
    numEnforced += 
      ContractsEnforcer_enforceClause(enforcer, ContractClause_INVARIANT,
        times.inv, times.routine) ? 1 : 0;
  }

  if (clauses & EnforcementClause_ASSERTS)
  {
    numEnforced += 
      ContractsEnforcer_enforceClause(enforcer, ContractClause_ASSERT,
        times.asrt, times.routine) ? 1 : 0;
  }

  ContractsEnforcer_logTrace(enforcer, times, NULL, NULL);

  return numEnforced;
}  /* checkRoutineClauses */


/**
 * Mimic running checking on an application/program but, instead of 
 * defining separate meaningful methods, simply run through enforcement
 * clause options.
 *
 * @param enforcer  [inout] The responsible contracts enforcer.
 * @param[in] iters Number of iterations through the clause(s).
 * @return          Number of enforced clauses.
 */ 
unsigned int
checkAppClauses(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    unsigned int           iters)  
{
  TimeEstimatesType times;
  unsigned int      i, ec, numEnforced = 0;

  ec            = (unsigned int) S_ENFORCEMENT_CLAUSE_MAX;
  times.pre     = 0;
  times.post    = 0;
  times.inv     = 0;
  times.asrt    = 0;
  times.routine = 5;

  for (i=0; i<iters; i++) 
  {
    numEnforced += checkRoutineClauses(enforcer, (EnforcementClauseEnum)ec, 
                                        times);

    ec = (ec > (unsigned int)S_ENFORCEMENT_CLAUSE_MIN) 
       ? ec-- : (unsigned int)S_ENFORCEMENT_CLAUSE_MAX;

    /* 
     * Provide variability for adaptive timing policies, though ignoring
     * the fact that different clauses for a given method call can have
     * different execution times.
     */
    times.pre++;
    times.post++;
    times.inv++;
    times.asrt++;
    times.routine += 5;
  }
  
  ContractsEnforcer_dumpStatistics(enforcer, "Completed checks");

  return numEnforced;
}  /* checkAppClauses */


/**
 * Test driver.  This routine instantiates each valid enforcer and
 * runs it against the test suite.
 */
int
main(int argc, char **argv)
{
  ContractsEnforcerType* enforcer = NULL;
  char                   *statsfile, *tracefile;
  unsigned int           max = 15;
  unsigned int           iFactor = 30;
  unsigned int           bad = 0, good = 0;
  unsigned int           numChecked = 0, passed = 0;
  unsigned int           policyValue = max, defaultPV = max;
  unsigned int           iterations = max*iFactor, defaultIters = iterations;
  int                    ec, ef, val;
  CONTRACTS_BOOL         checkDefault = CONTRACTS_FALSE;
  EnforcementClauseEnum  ece;
  EnforcementFrequencyEnum efe;

  /* 
   * The numbers of enforced clauses given in the table below MUST match what 
   * is actually obtained from default options.
   */
  static const unsigned int numEC = S_ENFORCEMENT_CLAUSE_MAX+1;
  static const unsigned int numEF = S_ENFORCEMENT_FREQUENCY_MAX+1;
  unsigned int total = numEC*numEF;
  unsigned int defaultEnforced[numEC][numEF] = 
  {
  /* NEVER, ALWAYS,   AF,   AT, PERIODIC, RANDOM */
    {    0,      0,    0,    0,        0,      0 }, /*  0=NONE */
    {    0,    900,    8,    8,       60,     60 }, /*  1=INVARIANTS */
    {    0,    450,    4,    4,       30,     30 }, /*  2=PRECONDITIONS */
    {    0,   1350,   12,   12,       90,     90 }, /*  3=INVPRE */
    {    0,    450,    4,    4,       30,     30 }, /*  4=POSTCONDITIONS */
    {    0,   1350,   12,   12,       90,     90 }, /*  5=INVPOST */
    {    0,    900,    8,    8,       60,     60 }, /*  6=PREPOST */
    {    0,   1800,   16,   16,      120,    120 }, /*  7=INVPREPOST */
    {    0,    450,    4,    4,       30,     30 }, /*  8=ASSERTIONS */
    {    0,   1350,   12,   12,       90,     90 }, /*  9=INVASRT */
    {    0,    900,    8,    8,       60,     60 }, /* 10=PREASRT */
    {    0,   1800,   16,   16,      120,    120 }, /* 11=INVPREASRT */
    {    0,    900,    8,    8,       60,     60 }, /* 12=POSTASRT */
    {    0,   1800,   16,   16,      120,    120 }, /* 13=INVPOSTASRT */
    {    0,   1350,   12,   12,       90,     90 }, /* 14=PREPOSTASRT */
    {    0,   2250,   20,   20,      150,    150 }  /* 15=ALL */
  };

  if (argc == 2) {
    policyValue = atoi(argv[1]);
    printf("\nAssuming the provided parameter (%s) is the policy value (%d).\n",
           argv[1], policyValue);
    iterations = policyValue*iFactor;
    printf("Number of iterations is defaulting to %d X policyValue=%d.\n\n", 
           iFactor, iterations);
  } else if (argc == 3) {
    policyValue = atoi(argv[1]);
    iterations = atoi(argv[2]);
  } else {
    printf("\nUSAGE: %s [<policy-value> [<iterations>]]", argv[0]);
    printf("\nwhere\n");
    printf("  <policy-value>  The overhead percent (adaptive frequencies),\n");
    printf("                    interval (periodic frequency), and window\n");
    printf("                    (random frequency).  [default=%d]\n", 
           policyValue);
    printf("  <iterations>    The number of check/enforcement iterations.\n");
    printf("                    This is the testing \"equivalent\" of the\n"); 
    printf("                    number of instrumented routines being\n");
    printf("                    simulated by the tests.  [default=%d]\n", 
           iterations);
    printf("\nProceeding with default options since none entered.\n");
  }

  setvbuf(stdout, NULL, _IONBF, 0);

  if ( (policyValue == defaultPV) && (iterations == defaultIters) ) {
    checkDefault = CONTRACTS_TRUE;
  }
  printf("\nRunning testContractEnforcer tests...\n");

  for (ec = (int)S_ENFORCEMENT_CLAUSE_MIN;
       ec <= (int)S_ENFORCEMENT_CLAUSE_MAX; ec++)
  {
    ece = (EnforcementClauseEnum) ec;

    for (ef = (int)S_ENFORCEMENT_FREQUENCY_MIN;
         ef <= (int)S_ENFORCEMENT_FREQUENCY_MAX; ef++)
    {
      efe = (EnforcementFrequencyEnum) ef;

      /*
       * Proceed with the test IF a suitable enforcer has been 
       * created.
       */
      statsfile = getFilename(ece, efe, FileType_STATISTICS, NULL);
      tracefile = getFilename(ece, efe, FileType_TRACE, NULL);
      enforcer = ContractsEnforcer_createEnforcer(ece, efe, policyValue,
                                                  statsfile, tracefile);
      if (enforcer != NULL) {
        good++;
        printf("  Case %s, %s: ", S_ENFORCEMENT_CLAUSE[ece],
             S_ENFORCEMENT_FREQUENCY[efe]);

        numChecked = checkAppClauses(enforcer, iterations);

        if (checkDefault) {
          if (numChecked == defaultEnforced[ec][ef]) {
            passed++;
            printf("  PASSED");
          } else {
            printf("  FAILED");
            printf(": %d vs. %d (expected)",numChecked,defaultEnforced[ec][ef]);
          }
        }

        ContractsEnforcer_free(enforcer);
      } else {
        bad++;
        printf("  FAILED: No enforcer");
      }
      printf("\n");

      if (statsfile != NULL) {
        free((void*)statsfile);
      }
      if (tracefile != NULL) {
        free((void*)tracefile);
      }
    }
  }

  printf("\n\nResults:\n  %d valid enforcers\n  %d invalid options", good, bad);
  if (checkDefault) {
    printf("\n  %d passed out of %d cases\n", passed, total);
    printf("\n\nTEST SUITE %s\n", (passed==total) ? "PASSED" : "FAILED");
  }

  return 0;
} /* main */ 
