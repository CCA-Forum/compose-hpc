/**
 * File:           ContractsEnforcer.c
 * Author:         T. Dahlgren
 * Created:        2012 May 11
 * Last Modified:  2012 November 12
 *
 *
 * @file
 * @section DESCRIPTION
 * Interface contract enforcement manager.
 *
 *
 * @section SOURCE
 * This implementation is based heavily on Babel's sidl_Enforcer and 
 * sidl_EnfPolicy.
 *
 *
 * @section COPYRIGHT
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Tamara Dahlgren <dahlgren1@llnl.gov>.
 * 
 * LLNL-CODE-473891.
 * All rights reserved.
 * 
 * This software is part of COMPOSE-HPC. See http://compose-hpc.sourceforge.net/
 * for details.  Please read the COPYRIGHT file for Our Notice and for the 
 * BSD License.
 */


#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>

#include "contracts.h"
#include "contractOptions.h"
#include "contractPrivateTypes.h"
#include "ContractsEnforcer.h"


#ifdef CONFENF_DEBUG
#define DEBUG_MESSAGE(MSG) printf("\nDEBUG: %s\n", (MSG));
#define DUMP_DEBUG_STATS(ENF, MSG) ContractsEnforcer_dumpStatistics(ENF, MSG);
#else
#define DEBUG_MESSAGE(MSG)
#define DUMP_DEBUG_STATS(ENF, MSG)
#endif /* DEBUG */


/**
 * Active "instance" data.
 */
const char*            pce_config_filename;
ContractsEnforcerType* pce_enforcer;
TimeEstimatesType      pce_def_times;


/*
 **********************************************************************
 * PRIVATE METHODS/ROUTINES
 **********************************************************************
 */

/**
 * INTERNAL USE ONLY.
 *
 * Create a basic enforcer that does NOT check any contracts.
 * 
 * @param clauses    Interface contract clause(s) to be checked.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
newBaseEnforcer(
  /* in */ EnforcementClauseEnum    clauses,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  static const char* l_timesLine = 
         "Pre Time (ms); Post Time (ms); Inv Time (ms); Routine Time (ms); ";

  ContractsEnforcerType* enforcer = (ContractsEnforcerType*)malloc(
                                      sizeof(ContractsEnforcerType));
  if (enforcer) 
  {
    DEBUG_MESSAGE("newBaseEnforcer: enforcer allocated")
    memset(enforcer, 0, sizeof(ContractsEnforcerType));
    enforcer->policy.clauses = clauses;
    enforcer->policy.frequency = EnforcementFrequency_NEVER;
    if ( (statsfile != NULL) && (strlen(statsfile) > 0) )
    {
      enforcer->stats = (EnforcementFileType*)malloc(
                                              sizeof(EnforcementFileType));
      if (enforcer->stats != NULL)
      {
        DEBUG_MESSAGE("newBaseEnforcer: stats allocated")
        enforcer->stats->fileName = strdup(statsfile);
        enforcer->stats->filePtr = fopen(statsfile, "w");
        if (enforcer->stats->filePtr != NULL) 
        {
          fprintf(enforcer->stats->filePtr, "%s%s%s%s\n",
            "Clauses; Frequency; Value; Timestamp; ",        /* Policy */
            "Requests; Requests Allowed; Countdown; Skip; ", /* State basics */
            l_timesLine,                                     /* Time totals */
            "Message"
            );
          fflush(enforcer->stats->filePtr);
        }
        else
        {
          printf("\nWARNING:  %s (%s).\n          %s\n",
                 "Cannot open enforcement statistics output file",
                 statsfile,
                 "Data will NOT be written.");
        }
      }
    }
    if ( (tracefile != NULL) && (strlen(tracefile) > 0) )
    {
      enforcer->trace = (EnforcementFileType*)malloc(
                                              sizeof(EnforcementFileType));
      if (enforcer->trace != NULL)
      {
        DEBUG_MESSAGE("newBaseEnforcer: trace allocated")
        enforcer->trace->fileName = strdup(tracefile);
        enforcer->trace->filePtr = fopen(tracefile, "w");
        if (enforcer->trace->filePtr != NULL) 
        {
          fprintf(enforcer->trace->filePtr, "%s%s%s\n",
                  "Name; ",        /* Trace identification */
                  l_timesLine,     /* Time estimates */
                  "Message"
            );
          fflush(enforcer->trace->filePtr);
        }
        else
        {
          printf("\nWARNING:  %s (%s).\n          %s\n",
                 "Cannot open enforcement trace output file",
                 tracefile,
                 "Traces will NOT be written.");
        }
      }
    }
    DUMP_DEBUG_STATS(enforcer, "newBaseEnforcer(): done")
  }

  return enforcer;
} /* newBaseEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create an invariants clause enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createInvEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_INVARIANTS, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_INVARIANTS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_INVARIANTS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_INVARIANTS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_INVARIANTS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_INVARIANTS, statsfile, 
                               tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createInvEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create a preconditions clause enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createPreEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_PRECONDITIONS, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_PRECONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_PRECONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_PRECONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_PRECONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_PRECONDITIONS, statsfile, 
                               tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createPreEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create an invariant and precondition clauses enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createInvPreEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_INVPRE, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_INVPRE, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_INVPRE, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_INVPRE, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_INVPRE, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_INVPRE, statsfile, tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createInvPreEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create a postcondition clause enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createPostEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_POSTCONDITIONS, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_POSTCONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_POSTCONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_POSTCONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_POSTCONDITIONS, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_POSTCONDITIONS, statsfile, 
                               tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createPostEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create an invariant and postcondition clauses enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createInvPostEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_INVPOST, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_INVPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_INVPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_INVPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_INVPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_INVPOST, statsfile, tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createInvPostEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create a precondition and precondition clauses enforcer based on the 
 * specified frequency.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createPrePostEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_PREPOST, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_PREPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                 EnforcementClause_PREPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                 EnforcementClause_PREPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                 EnforcementClause_PREPOST, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_PREPOST, statsfile, tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createPrePostEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Create an all interface contract clause enforcer.
 *
 * @param frequency  Frequency of checking the clauses encountered.
 * @param value      The policy value option, when appropriate.
 * @param statsfile  [Optional] Name of the file to output enforcement data.
 * @param tracefile  [Optional] Name of the file to output enforcement traces.
 * @return           Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
createAllEnforcer(
  /* in */ EnforcementFrequencyEnum frequency, 
  /* in */ unsigned int             value,
  /* in */ const char*              statsfile,
  /* in */ const char*              tracefile)
{
  ContractsEnforcerType* enforcer = NULL;

  switch (frequency)
  {
  case EnforcementFrequency_ALWAYS:
    enforcer = ContractsEnforcer_setEnforceAll(
                 EnforcementClause_ALL, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_FIT:
    enforcer = ContractsEnforcer_setEnforceAdaptiveFit(
                 EnforcementClause_ALL, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_ADAPTIVE_TIMING:
    enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(
                EnforcementClause_ALL, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_PERIODIC:
    enforcer = ContractsEnforcer_setEnforcePeriodic(
                EnforcementClause_ALL, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_RANDOM:
    enforcer = ContractsEnforcer_setEnforceRandom(
                EnforcementClause_ALL, value, statsfile, tracefile);
    break;
  case EnforcementFrequency_NEVER:
    /* TBD: Is this case even worth doing? */
    enforcer = newBaseEnforcer(EnforcementClause_ALL, statsfile, tracefile);
    enforcer->policy.value = value;
    break;
  default:
    /* Not a valid combination */
    break;
  } /* frequency */

  return enforcer;
} /* createAllEnforcer */


/**
 * INTERNAL USE ONLY.
 *
 * Resets the countdown, if applicable, based on enforcement options.
 * 
 * @param enforcer  Responsible enforcer.
 */
void
resetEnforcementCountdown(
  /* inout */ ContractsEnforcerType* enforcer)
{
  unsigned int rcd;

  if (enforcer)
  {
    DEBUG_MESSAGE("resetEnforcementCountdown(): begin")
    DUMP_DEBUG_STATS(enforcer, "resetEnforcementCountdown(): begin")
    if (enforcer->policy.frequency == EnforcementFrequency_PERIODIC) 
    {
      enforcer->data.countdown = enforcer->policy.value;
      enforcer->data.skip      = 0;
    }
    else if (enforcer->policy.frequency == EnforcementFrequency_RANDOM)
    {
      rcd = (int32_t)(ceil( ((double)rand()/(double)RAND_MAX)
                          * ((double)(enforcer->policy.value)) ) );
      enforcer->data.countdown = enforcer->data.skip + rcd;
      enforcer->data.skip      = enforcer->policy.value - rcd;
    }
    DUMP_DEBUG_STATS(enforcer, "resetEnforcementCountdown(): end")
    DEBUG_MESSAGE("resetEnforcementCountdown(): end")
  }

  return;
} /* resetEnforcementCountdown */


/**
 * INTERNAL USE ONLY.
 *
 * Determine if it is time to check a contract clause based on enforcement
 * options.  Adjusts enforcement state as needed for some policies.
 * 
 * @param enforcer    Responsible enforcer.
 * @param clauseTime   The time it is estimated to take to check the clause.
 * @param routineTime  The time it is estimated to take to execute the routine
 *                       body.
 * @return            CONTRACTS_TRUE if time to check a clause; otherwise,
 *                      CONTRACTS_FALSE.
 */
CONTRACTS_BOOL
timeToCheckClause(
  /* inout */ ContractsEnforcerType* enforcer,
  /* in */    uint64_t               clauseTime,
  /* in */    uint64_t               routineTime)
{
  CONTRACTS_BOOL checkIt = CONTRACTS_FALSE;
  uint64_t       enforceTotal, routinesTotal;
  double         limit;

  if (enforcer)
  {
    DEBUG_MESSAGE("timeToCheckClause(): begin (with enforcer)")
    switch (enforcer->policy.frequency)
    {
    case EnforcementFrequency_ALWAYS:
      checkIt = CONTRACTS_TRUE;
      break;
    case EnforcementFrequency_ADAPTIVE_FIT:
      enforceTotal = clauseTime + enforcer->data.checksTime;
      limit = (double)(routineTime) * (double)enforcer->policy.value/100.0;
      if ((double)clauseTime <= limit)
      {
        checkIt = CONTRACTS_TRUE;
      }
      break;
    case EnforcementFrequency_ADAPTIVE_TIMING:
      limit = (double)(routineTime) * (double)enforcer->policy.value/100.0;
      if ((double)clauseTime <= limit)
      {
        checkIt = CONTRACTS_TRUE;
      } 
      else if (clauseTime <= 1)
      {
        enforceTotal = clauseTime + enforcer->data.checksTime;
        limit = (double)(enforcer->data.total.routine) 
              * (double)enforcer->policy.value/100.0;
          checkIt = CONTRACTS_TRUE;
      }
      break;
    case EnforcementFrequency_PERIODIC:
    case EnforcementFrequency_RANDOM:
      if (enforcer->data.countdown > 1)
      {
        DUMP_DEBUG_STATS(enforcer, "timeToCheckClause(): begin")
        (enforcer->data.countdown)--;
        DUMP_DEBUG_STATS(enforcer, "timeToCheckClause(): end")
      }
      else
      {
        checkIt = CONTRACTS_TRUE;
        resetEnforcementCountdown(enforcer);
      }
      break;
    default:
      /* Don't check by default */
      break;
    }
    DEBUG_MESSAGE("timeToCheckClause(): end (with enforcer)")
  }

  return checkIt;
} /* timeToCheckClause */


/*
 **********************************************************************
 * PUBLIC METHODS/ROUTINES
 **********************************************************************
 */


/**
 * FOR APPLICATION USE.  
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
  /* in */ const char* configfile)
{
  FILE* cfPtr = NULL;

  if (configfile == NULL)
  {
    DEBUG_MESSAGE("ContractsEnforcer_initialize(): No config file given")
    pce_config_filename = configfile;
    memset(&pce_def_times, 0, sizeof(TimeEstimatesType));
    pce_enforcer = ContractsEnforcer_setEnforceAll(EnforcementClause_ALL, 
      NULL, NULL);
  }
  else if (strlen(configfile) > 0)
  {
    DEBUG_MESSAGE("ContractsEnforcer_initialize(): Config file given")
#if TBD_PCE_CONFIGURATION
    /*
     * @todo Review potential runtime location issues with configuration file. 
     */
    pce_config_filename = strdup(configfile);
    cfPtr = fopen(configfile, "r");
    if (cfPtr!= NULL) 
    {
      /*
       * Read entries from the configuration file to set the following:
       *
       * - Enforcement clause(s);
       * - Enforcement frequency;
       * - Enforcement value (see frequency options);
       * - Statistics (output) file;
       * - Trace (output) file; and
       * - Default clause and routine timing data: pce_def_times.
       */

      fclose(cfPtr);
    }
#endif /* TBD_PCE_CONFIGURATION */
    printf("\nFATAL: %s %s\n",
           "Contract enforcement initialization from configuration file",
           "not yet supported.");
    exit(1);
  }
  else
  {
    printf("\nFATAL: %s %s\n",
           "Contract enforcement initialization with non-NULL configuration",
           "file requires a non-empty file name.");
    exit(1);
  }

  return;
}  /* ContractsEnforcer_initialize */


/**
 * FOR APPLICATION/AUTOMATED INSTRUMENTATION USE.  
 *
 * Finalize the global enforcer, releasing memory and performing associated
 * clean up.
 */
void
ContractsEnforcer_finalize(void)
{
  DEBUG_MESSAGE("ContractsEnforcer_finalize(): begin")
  ContractsEnforcer_free(pce_enforcer);

  if (pce_config_filename != NULL) 
  {
    free((void*)pce_config_filename);
  }

  /* Just make SURE everything is cleared. */
  pce_enforcer = NULL;
  pce_config_filename = NULL;
  memset(&pce_def_times, 0, sizeof(TimeEstimatesType));

  DEBUG_MESSAGE("ContractsEnforcer_finalize(): end")
  return;
}  /* ContractsEnforcer_finialize */



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
  /* in */ const char*              tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_createEnforcer(): begin")
  ContractsEnforcerType* enforcer = NULL;

  switch (clauses)
  {
  case EnforcementClause_NONE:
    if (frequency == EnforcementFrequency_NEVER)
    {
      enforcer = ContractsEnforcer_setEnforceNone();
    }
    else if (frequency == EnforcementFrequency_ALWAYS)
    {
      enforcer = ContractsEnforcer_setEnforceAll(clauses, statsfile, tracefile);
    }
    else if (frequency == EnforcementFrequency_ADAPTIVE_FIT)
    {
      enforcer = ContractsEnforcer_setEnforceAdaptiveFit(clauses, 0, statsfile,
                                                         tracefile);
    }
    else if (frequency == EnforcementFrequency_ADAPTIVE_TIMING)
    {
      enforcer = ContractsEnforcer_setEnforceAdaptiveTiming(clauses, 0, 
                   statsfile, tracefile);
    }
    else if (frequency == EnforcementFrequency_PERIODIC)
    {
      enforcer = ContractsEnforcer_setEnforcePeriodic(clauses, 0, statsfile, 
                                                      tracefile);
    }
    else if (frequency == EnforcementFrequency_RANDOM)
    {
      enforcer = ContractsEnforcer_setEnforceRandom(clauses, 0, statsfile, 
                                                    tracefile);
    }
    else 
    {
      printf("\nERROR: Unrecognized/unsupported enforcement frequency %d\n",
             frequency);
    }
    break;
  case EnforcementClause_INVARIANTS:
    enforcer = createInvEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_PRECONDITIONS:
    enforcer = createPreEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_INVPRE:
    enforcer = createInvPreEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_POSTCONDITIONS:
    enforcer = createPostEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_INVPOST:
    enforcer = createInvPostEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_PREPOST:
    enforcer = createPrePostEnforcer(frequency, value, statsfile, tracefile);
    break;
  case EnforcementClause_ALL:
    enforcer = createAllEnforcer(frequency, value, statsfile, tracefile);
    break;
  default:
    printf("ERROR:  Unrecognized enforcement clause option.\n");
    break;
  } /* clauses */

  DEBUG_MESSAGE("ContractsEnforcer_createEnforcer(): end")
  return enforcer;
} /* ContractsEnforcer_createEnforcer */


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
  /* in */ const char*           tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAll(): begin")
  ContractsEnforcerType* enforcer = 
    newBaseEnforcer(clauses, statsfile, tracefile);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_ALWAYS;
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforceAll(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAll(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforceAll */


/**
 * FOR APPLICATION USE.  
 *
 * Create an enforcer whose policy is to NEVER check any contract clauses.
 * 
 * @return  Pointer to the initialized enforcer, if successful.
 */
ContractsEnforcerType*
ContractsEnforcer_setEnforceNone(void)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforceNone(): begin")
  ContractsEnforcerType* enforcer = newBaseEnforcer(EnforcementClause_NONE, 
                                                    NULL, NULL);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_NEVER;
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforceNone(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforceNone(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforceNone */


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
  /* in */ const char*            tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforcePeriodic(): begin")
  ContractsEnforcerType* enforcer = 
    newBaseEnforcer(clauses, statsfile, tracefile);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_PERIODIC;
    enforcer->policy.value = interval;
    resetEnforcementCountdown(enforcer);
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforcePeriodic(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforcePeriodic(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforcePeriodic */


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
  /* in */ const char*            tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforceRandom(): begin")
  ContractsEnforcerType* enforcer = 
    newBaseEnforcer(clauses, statsfile, tracefile);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_RANDOM;
    enforcer->policy.value = window;
    resetEnforcementCountdown(enforcer);
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforceRandom(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforceRandom(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforceRandom */


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
  /* in */ const char*            tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAdaptiveFit(): begin")
  ContractsEnforcerType* enforcer = 
    newBaseEnforcer(clauses, statsfile, tracefile);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_ADAPTIVE_FIT;
    if (limit < 1)
    {
      enforcer->policy.value = 1;
    }
    else if (limit > 99)
    {
      enforcer->policy.value = 99;
    }
    else
    {
      enforcer->policy.value = limit;
    }
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforceAdaptiveFit(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAdaptiveFit(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforceAdaptiveFit */


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
  /* in */ const char*            tracefile)
{
  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAdaptiveTiming(): begin")
  ContractsEnforcerType* enforcer = 
    newBaseEnforcer(clauses, statsfile, tracefile);
  if (enforcer) {
    enforcer->policy.frequency = EnforcementFrequency_ADAPTIVE_TIMING;
    if (limit < 1)
    {
      enforcer->policy.value = 1;
    }
    else if (limit > 99)
    {
      enforcer->policy.value = 99;
    }
    else
    {
      enforcer->policy.value = limit;
    }
  }
  DUMP_DEBUG_STATS(enforcer, "setEnforceAdaptiveTiming(): done")

  DEBUG_MESSAGE("ContractsEnforcer_setEnforceAdaptiveTiming(): end")
  return enforcer;
} /* ContractsEnforcer_setEnforceAdaptiveTiming */


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
  /* in */ const char*            msg)
{
  time_t      currTime;
  char*       timeStr;
  const char* cmt;

  DEBUG_MESSAGE("ContractsEnforcer_dumpStatistics(): begin")
  if ( (enforcer != NULL) && (enforcer->stats != NULL) )
  {
    cmt = (msg != NULL) ? msg : "";
    currTime = time(NULL);
    timeStr  = ctime(&currTime);  /* Static so do NOT free() */
    timeStr[24] = '\0';           /* Only need 1st 24 characters */

    if (enforcer->stats->filePtr != NULL) 
    {
      /*
        "Clauses; Frequency; Value; Timestamp; ",          Policy
        "Requests; Requests Allowed; Countdown; Skip; ",   State basics
        "Pre Time (ms); Post Time (ms); Inv Time (ms); Routine Time (ms);", 
        "Message"
       */
      fprintf(enforcer->stats->filePtr, 
              "%s; %s; %d; %s; %ld; %ld; %d; %d; %ld; %ld; %ld; %ld; %s\n",
              S_ENFORCEMENT_CLAUSE[enforcer->policy.clauses],
              S_ENFORCEMENT_FREQUENCY[enforcer->policy.frequency],
              enforcer->policy.value,
              timeStr,
              enforcer->data.requests,
              enforcer->data.allowed,
              enforcer->data.countdown,
              enforcer->data.skip,
              enforcer->data.total.pre, 
              enforcer->data.total.post, 
              enforcer->data.total.inv, 
              enforcer->data.total.routine,
              cmt);
      fflush(enforcer->stats->filePtr);
    }
#ifdef CONFENF_DEBUG
    else
    {
      /*
       * WARNING:  The following should be kept in sync with the output above.
       */
      printf("%s; %s; %d; %s; %ld; %ld; %d; %d; %ld; %ld; %ld; %ld; %s\n",
             S_ENFORCEMENT_CLAUSE[enforcer->policy.clauses],
             S_ENFORCEMENT_FREQUENCY[enforcer->policy.frequency],
             enforcer->policy.value,
             timeStr,
             enforcer->data.requests,
             enforcer->data.allowed,
             enforcer->data.countdown,
             enforcer->data.skip,
             enforcer->data.total.pre, 
             enforcer->data.total.post, 
             enforcer->data.total.inv, 
             enforcer->data.total.routine,
             cmt);
    }
#endif
  }

  DEBUG_MESSAGE("ContractsEnforcer_dumpStatistics(): end")
  return;
} /* ContractsEnforcer_dumpStatistics */


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
  /* inout */ ContractsEnforcerType* enforcer)
{
  DEBUG_MESSAGE("ContractsEnforcer_free(): begin")
  if (enforcer) 
  {
    if (enforcer->stats != NULL)
    {
      if (enforcer->stats->filePtr)
      {
        fclose(enforcer->stats->filePtr);
      }
      if (enforcer->stats->fileName)
      {
        free(enforcer->stats->fileName);
      }
      free(enforcer->stats);
    }
    if (enforcer->trace != NULL)
    {
      if (enforcer->trace->filePtr)
      {
        fclose(enforcer->trace->filePtr);
      }
      if (enforcer->trace->fileName)
      {
        free(enforcer->trace->fileName);
      }
    }
  }
  enforcer = NULL;

  DEBUG_MESSAGE("ContractsEnforcer_free(): end")
  return;
} /* ContractsEnforcer_free */


/**
 * TBD/ToDo:  How should this work IF decide to actually include it?
 *
 * FOR APPLICATION USE.  
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
  /* in */ const char*            msg)
{
  const char* nm = (name != NULL) ? name : "TRACE";
  const char* cmt = (msg != NULL) ? msg : "";

  DEBUG_MESSAGE("ContractsEnforcer_logTrace(): begin")
  if ( (enforcer != NULL) && (enforcer->trace != NULL) )
  {
    if (enforcer->trace->filePtr != NULL) 
    {
      /*
         "Name; ",        Trace identification
         "Pre Time (ms); Post Time (ms); Inv Time (ms); Routine Time (ms);", 
         "Message"
      */
      fprintf(enforcer->trace->filePtr, "%s; %ld; %ld; %ld; %ld; %s\n",
              nm, times.pre, times.post, times.inv, times.routine, cmt);
      fflush(enforcer->trace->filePtr);
    }
  }

  DEBUG_MESSAGE("ContractsEnforcer_logTrace(): end")
  return;
} /* ContractsEnforcer_logTrace */


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
  /* in */    CONTRACTS_BOOL         firstForCall)
{
  CONTRACTS_BOOL checkIt = CONTRACTS_FALSE;

  DEBUG_MESSAGE("ContractsEnforcer_enforceClause(): begin")
  if (enforcer)
  {
    (enforcer->data.requests)++;
    if (firstForCall)
    {
      (enforcer->data.total.routine) += routineTime;
    }

    if (enforcer->policy.clauses & clause) 
    {
      checkIt = timeToCheckClause(enforcer, clauseTime, routineTime);

      if (checkIt)
      {
        DUMP_DEBUG_STATS(enforcer, "enforceClause(): begin")
        (enforcer->data.allowed)++;
        switch (clause)
        {
        case ContractClause_INVARIANT:
          (enforcer->data.total.inv) += clauseTime;
          break;
        case ContractClause_PRECONDITION:
          (enforcer->data.total.pre) += clauseTime;
          break;
        case ContractClause_POSTCONDITION:
          (enforcer->data.total.post) += clauseTime;
          break;
        default:
          /* Nothing to do here */
          break;
        }

        /* 
         * Cache the time for speed (versus space) trade-off, clearly
         * assuming we'll NEVER reach the default case above.
         */
        (enforcer->data.checksTime) += clauseTime;
        DUMP_DEBUG_STATS(enforcer, "enforceClause(): end")
      }
    }
  }

  DEBUG_MESSAGE("ContractsEnforcer_enforceClause(): end")
  return checkIt;
} /* ContractsEnforcer_enforceClause */
