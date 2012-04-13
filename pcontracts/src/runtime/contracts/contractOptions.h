/*
 * File:          contractOptions.h
 * Description:   Interface contract enforcement options
 * Source:        Borrowed heavily from Babel's SIDL enforcement options
 * 
 * Copyright (c) 2012, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 */

#ifndef contractOptions_h
#define contractOptions_h

#ifdef __cplusplus
extern "C" {
#endif

/*
 **********************************************************************
 *                   CONTRACT ENFORCEMENT OPTION TYPES                *
 **********************************************************************
 */

/* 
 * Contract clause types. 
 */
typedef enum ContractClause__enum {
  ContractClause_INVARIANT     = 1,
  ContractClause_PRECONDITION  = 2,
  ContractClause_POSTCONDITION = 4
} ContractClauseEnum;

/*
 * Names corresponding to (and indexed by) ContractClauseEnum.
 */
static const char* S_CONTRACT_CLAUSE[5] = {
  "undefined",
  "Invariant",
  "Precondition",
  "undefined",
  "Postcondition"
};
static const unsigned int S_CONTRACT_CLAUSE_MIN_IND = 0;
static const unsigned int S_CONTRACT_CLAUSE_MAX_IND = 4;


/* 
 * Enforcement clause options.
 */
typedef enum EnforcementClause__enum {
  EnforcementClause_INVARIANTS    = ContractClause_INVARIANT, /* 1 */
  EnforcementClause_PRECONDITIONS = ContractClause_PRECONDITION, /* 2 */
  EnforcementClause_INVPRE        = ContractClause_INVARIANT 
                                  | ContractClause_PRECONDITION, /* 3 */
  EnforcementClause_POSTCONDITION = ContractClause_POSTCONDITION, /* 4 */
  EnforcementClause_INVPOST       = ContractClause_INVARIANT 
                                  | ContractClause_POSTCONDITION, /* 5 */
  EnforcementClause_PREPOST       = ContractClause_PRECONDITION
                                  | ContractClause_POSTCONDITION, /* 6 */
  EnforcementClause_ALL           = ContractClause_INVARIANT
                                  | ContractClause_PRECONDITION
                                  | ContractClause_POSTCONDITION /* 7 */
} EnforcementClauseEnum;
static const EnforcementClauseEnum S_ENFORCEMENT_CLAUSE_MIN 
                                   = EnforcementClause_INVARIANTS;
static const EnforcementClauseEnum S_ENFORCEMENT_CLAUSE_MAX
                                   = EnforcementClause_ALL;

/* 
 * Contract clause enforcement frequency.
 */
typedef enum EnforcementFrequency__enum {
  EnforcementFrequency_NEVER       = 0,
  EnforcementFrequency_ALWAYS      = 1,
  EnforcementFrequency_ADAPTIVE    = 2,
  EnforcementFrequency_PERIODIC    = 3,
  EnforcementFrequency_RANDOM      = 4
} EnforcementFrequencyEnum;
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MIN 
                                      = EnforcementFrequency_NEVER;
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MAX
                                      = EnforcementFrequency_RANDOM;

/*
 * Names corresponding to (and indexed by) EnforcementFrequencyEnum.
 */
static const char* S_ENFORCEMENT_FREQUENCY[5] = {
  "Never",
  "Always",
  "Adaptive",
  "Periodic",
  "Random"
};
static const unsigned int S_ENFORCEMENT_FREQUENCY_MIN_IND = 0;
static const unsigned int S_ENFORCEMENT_FREQUENCY_MAX_IND = 4;


#if 0
/*
 * TBD/TODO:  Is this capability going to be retained?
 *
 * Contract enforcement tracing levels.  Enforcement traces rely on
 * automatically inserted runtime timing.
 */
typedef enum ContractTrace__enum {
  ContractTrace_NONE     = 0,
  ContractTrace_CORE     = 1,
  ContractTrace_BASIC    = 2,
  ContractTrace_OVERHEAD = 3
} ContractTraceEnum;
static const ContractTraceEnum S_CONTRACT_TRACE_MIN = ContractTrace_NONE;
static const ContractTraceEnum S_CONTRACT_TRACE_MAX = ContractTrace_OVERHEAD;

/*
 * Names corresponding to (and indexed by) ContractTraceEnum.
 */
static const char* S_CONTRACT_TRACE[4] = {
  "None",
  "Core",
  "Basic",
  "Overhead"
};
static const unsigned int S_CONTRACT_TRACE_MIN_IND = 0;
static const unsigned int S_CONTRACT_TRACE_MAX_IND = 3;
#endif 


#ifdef __cplusplus
}
#endif

#endif /* contractOptions_h */
