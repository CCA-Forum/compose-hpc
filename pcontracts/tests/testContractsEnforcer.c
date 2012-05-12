/*
 * File:         testContractsEnforcer.c
 * Description:  Test suite for ContractsEnforcer
 *
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the lawrence Livermore National Laboratory.
 * All rights reserved.
 */
#include <stdio.h>
#include "ContractsEnforcer.h"

static ContractsEnforcer* s_enforcer = NULL;

/**
 * Returns the desired contracts enforcer.
 *
 * @param clauses   The desired enforcement clause option.
 * @param frequency The desired enforcement frequency option.
 * @param value     The policy value option, when appropriate.
 * @returns         A pointer to the contracts enforcer.
 */
ContractsEnforcer*
createEnforcer(
  /* in */ EnforcementClauseEnum    clauses, 
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value) 
{
  ContractsEnforcer* enforcer = NULL;

  /*
   * Ack!  Need to simplify similar clause-frequency -> call combinations.
   */
  switch (clauses)
  {
  case EnforcementClause_NONE:
    if (frequency == EnforcementFrequency_NEVER)
    {
      enforcer = ContractsEnforcer_setEnforceNone();
    }
    else if (frequency == EnforcementFrequency_ALWAYS)
    {
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_None-stats.csv",
                    "testContractsEnforcer_All_None-trace.csv");
    }
    /* else ignore any other combinations */
    break;
  case EnforcementClause_INVARIANTS:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_Inv-stats.csv",
                    "testContractsEnforcer_All_Inv-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_Inv-stats.csv",
                    "testContractsEnforcer_AdF_Inv-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_Inv-stats.csv",
                    "testContractsEnforcer_AdT_Inv-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_Inv-stats.csv",
                    "testContractsEnforcer_Per_Inv-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_Inv-stats.csv",
                    "testContractsEnforcer_Ran_Inv-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_PRECONDITIONS:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_Pre-stats.csv",
                    "testContractsEnforcer_All_Pre-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_Pre-stats.csv",
                    "testContractsEnforcer_AdF_Pre-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_Pre-stats.csv",
                    "testContractsEnforcer_AdT_Pre-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_Pre-stats.csv",
                    "testContractsEnforcer_Per_Pre-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_Pre-stats.csv",
                    "testContractsEnforcer_Ran_Pre-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_INVPRE:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_IPr-stats.csv",
                    "testContractsEnforcer_All_IPr-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_IPr-stats.csv",
                    "testContractsEnforcer_AdF_IPr-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_IPr-stats.csv",
                    "testContractsEnforcer_AdT_IPr-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_IPr-stats.csv",
                    "testContractsEnforcer_Per_IPr-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_IPr-stats.csv",
                    "testContractsEnforcer_Ran_IPr-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_POSTCONDITIONS:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_Pos-stats.csv",
                    "testContractsEnforcer_All_Pos-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_Pos-stats.csv",
                    "testContractsEnforcer_AdF_Pos-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_Pos-stats.csv",
                    "testContractsEnforcer_AdT_Pos-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_Pos-stats.csv",
                    "testContractsEnforcer_Per_Pos-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_Pos-stats.csv",
                    "testContractsEnforcer_Ran_Pos-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_INVPOST:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_IPo-stats.csv",
                    "testContractsEnforcer_All_IPo-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_IPo-stats.csv",
                    "testContractsEnforcer_AdF_IPo-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_IPo-stats.csv",
                    "testContractsEnforcer_AdT_IPo-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_IPo-stats.csv",
                    "testContractsEnforcer_Per_IPo-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_IPo-stats.csv",
                    "testContractsEnforcer_Ran_IPo-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_PREPOST:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_PnP-stats.csv",
                    "testContractsEnforcer_All_PnP-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_PnP-stats.csv",
                    "testContractsEnforcer_AdF_PnP-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_PnP-stats.csv",
                    "testContractsEnforcer_AdT_PnP-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_PnP-stats.csv",
                    "testContractsEnforcer_Per_PnP-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_PnP-stats.csv",
                    "testContractsEnforcer_Ran_PnP-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  case EnforcementClause_ALL:
    switch (frequency)
    {
    case EnforcementFrequency_ALWAYS:
      enforcer = ContractsEnforcer_setEnforceAll(clauses,
                    "testContractsEnforcer_All_All-stats.csv",
                    "testContractsEnforcer_All_All-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, value,
                    "testContractsEnforcer_AdF_All-stats.csv",
                    "testContractsEnforcer_AdF_All-trace.csv");
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, value,
                    "testContractsEnforcer_AdT_All-stats.csv",
                    "testContractsEnforcer_AdT_All-trace.csv");
      break;
    case EnforcementFrequency_PERIODIC:
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, value,
                    "testContractsEnforcer_Per_All-stats.csv",
                    "testContractsEnforcer_Per_All-trace.csv");
      break;
    case EnforcementFrequency_RANDOM:
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, value,
                    "testContractsEnforcer_Ran_All-stats.csv",
                    "testContractsEnforcer_Ran_All-trace.csv");
      break;
    default:
      /* Irrelevant combination */
      break;
    } /* frequency */
    break;
  default:
    printf("ERROR:  Unrecognized enforcement clause option.\n");
    break;
  } /* clauses */

  return enforcer;
} /* createEnforcer */


// TBD/ToDo:  Need to write one or more routines with invariants, 
// preconditions, and postconditions executed within loop(s)

int
main(int argc, char **argv)
{
  unsigned int count = 0;
  unsigned int policyValue = 100;
  unsigned int iterations = 100;
  int          val;

  printf(
    "\nUSAGE: %s [<policy-value> [<iterations>]], each defaulting to 100\n",
    argv[0]);
  if (argc == 2)
  {
    policyValue = atoi(argv[1]);
  } 
  else if (argc == 3)
  {
    policyValue = atoi(argv[1]);
    iterations = atoi(argv[2]);
  }

  for (EnforcementClauseEnum ec = S_ENFORCEMENT_CLAUSE_MIN;
       ec <= S_ENFORCEMENT_CLAUSE_MAX; ec++)
  {
    for (EnforcementFrequencyEnum ef = S_ENFORCEMENT_FREQUENCY_MIN;
         ef <= S_ENFORCEMENT_FREQUENCY_MAX; ef++)
    {
      s_enforcer = createEnforcer(ec, ef, policyValue);

      /*
       * Proceed with the test IF a suitable enforcer has been 
       * created.
       */
      if (s_enforcer != NULL)
      {
        count++;

        // TBD/ToDo:  Call the code...after writing it!

        ContractsEnforcer_free(s_enforcer);
      }
    }
  }

  printf("\nTested %d enforcers\n", count);

  return 0;
} /* main */ 
