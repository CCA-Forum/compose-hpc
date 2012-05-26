/**
 * File:  testContractsEnforcer.c
 *
 * @file
 * @section DESCRIPTION
 * Test suite for ContractsEnforcer.
 *
 * @section LICENSE
 * TBD
 *
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ContractsEnforcer.h"

/**
 * Abbreviated names corresponding to (and indexable by) EnforcementClauseEnum.
 *
 * NOTE:  These names MUST be kept in sync with EnforcementClauseEnum values.
 */
static const char* S_FILE_CLAUSES[8] = {
  "Non",
  "Inv",
  "Pre",
  "IPr",
  "Pos"
  "IPo",
  "PPo",
  "IPP"
};

/**
 * Abbreviated names corresponding to (and indexable by) 
 * EnforcementFrequencyEnum.
 *
 * NOTE:  These names MUST be kept in sync with EnforcementFrequencyEnum values.
 */
static const char* S_FILE_FREQUENCY[6] = {
  "Nev",
  "All",
  "AdF",
  "AdT",
  "Per",
  "Ran"
};

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
 * @param clauses    Clause(s) to be checked.
 * @param frequency  Frequency of checking.
 * @param fileType   Type of file being created.
 * @param ext        Desired file extension [Default=csv].
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
  char* pre = "testContractsEnforcer";
  char* clauseStr = S_FILENAME_CLAUSES[clauses];
  char* freqStr = S_FILENAME_FREQUENCY[frequency];
  char* typeStr = S_FILE_TYPE[fileType];
  char* extStr = strlen(ext) > 0 ? ext : "csv";

  int len = strlen(pre) + strlen(clauseStr) + strlen(freqStr)
          + strlen(typeStr) + strlen(extStr) + 5;
  fn = (char*)malloc(len*sizeof(char));
  if (fn != NULL) {
    sprintf(fn, "%s-%s-%s-%s.%s", pre, clauseStr, freqStr, typeStr, extStr);
  }
  return fn;
} /* getFilename */


/*
 * TBD/ToDo:  Need to write one or more routines that call and, if 
 * appropriate, check results of calls to "public" ContractsEnforcer 
 * routines.  These should be in loops in order to ensure sampling-
 * based enforcement policies actually engage.
 *
 * ContractsEnforcer_enforceClause(enforcer, clause, clauseTime, 
 *                                 routineTime, firstForCall)
 * ContractsEnforcer_dumpStatistics(enforcer, msg)
 * ContractsEnforcer_logTrace(enforcer, times, name, msg)
 */


/**
 * Test driver.  This routine instantiates each valid enforcer and
 * runs it against the test suite.
 */
int
main(int argc, char **argv)
{
  ContractsEnforcer* enforcer = NULL;
  unsigned int max = 100;
  unsigned int bad = 0;
  unsigned int good = 0;
  unsigned int policyValue = max;
  unsigned int iterations = max;
  int          val;

  if (argc == 2) {
    policyValue = atoi(argv[1]);
    printf("\nAssuming the provided parameter (%d) is the policy value.\n",
           argv[0]);
    printf("The number of iterations is defaulting to %d.\n\n", max);
  } else if (argc == 3) {
    policyValue = atoi(argv[1]);
    iterations = atoi(argv[2]);
  } else {
    printf(
      "\nUSAGE: %s [<policy-value> [<iterations>]], each defaulting to %d\n",
      argv[0], max);
  }

  for (EnforcementClauseEnum ec = S_ENFORCEMENT_CLAUSE_MIN;
       ec <= S_ENFORCEMENT_CLAUSE_MAX; ec++)
  {
    for (EnforcementFrequencyEnum ef = S_ENFORCEMENT_FREQUENCY_MIN;
         ef <= S_ENFORCEMENT_FREQUENCY_MAX; ef++)
    {
      char* statsFile = getFilename(ec, ef, FileType_STATISTICS, NULL);
      char* traceFile = getFilename(ec, ef, FileType_TRACE, NULL);

      enforcer = ContractsEnforcer_createEnforcer(ec, ef, policyValue,
                   statsFile, traceFile);

      /*
       * Proceed with the test IF a suitable enforcer has been 
       * created.
       */
      if (enforcer != NULL) {
        good++;

        // TBD/ToDo:  Call routines exercising the "public" methods.

        ContractsEnforcer_free(enforcer);
      } else {
        bad++;
        printf("\nFailed to create enforcer for %s and %s\n",
               S_ENFORCEMENT_CLAUSE[clauses], 
               S_ENFORCEMENT_FREQUENCY[frequency]);
      }

      if (statsFile != NULL) {
        free(statsFile);
      }
      if (traceFile != NULL) {
        free(traceFile);
      }
    }
  }

  printf("\n\nResults:\n  %d valid enforcers\n  %d invalid options", good, bad);

  return 0;
} /* main */ 
