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

/**
 * ------------------------------------------------------------------
 * Contract clause types. 
 *
 * WARNING:  ContractClauseEnum and S_CONTRACT_CLAUSE _must_ be kept 
 *   in sync.  That is, any changes to one must correspond to changes
 *   in the other.
 * ------------------------------------------------------------------
 */
typedef enum ContractClause__enum {
  ContractClause_INVARIANT     = 1,
  ContractClause_PRECONDITION  = 2,
  ContractClause_POSTCONDITION = 4
} ContractClauseEnum;

/**
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


/**
 * ------------------------------------------------------------------
 * Enforcement clause options.
 *
 * WARNING:  EnforcementClauseEnum and S_ENFORCEMENT_CLAUSE _must_ be 
 *   kept in sync.  That is, any changes to one must correspond to 
 *   changes in the other.
 * ------------------------------------------------------------------
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

/**
 * Names corresponding to (and indexed by) EnforcementClauseEnum.
 *
 * NOTE:  While the names shown below could be derived from the
 *   actual clauses at runtime, it was decided to maintain them
 *   here for consistency with the other enforcement-related
 *   enumerations.
 */
static const char* S_ENFORCEMENT_CLAUSE[8] = {
  "undefined",
  "Invariant",
  "Precondition",
  "Invariant-Precondition",
  "Postcondition"
  "Invariant-Postcondition",
  "Precondition-Postcondition",
  "Invariant-Precondition-Postcondition"
};
static const unsigned int S_ENFORCEMENT_CLAUSE_MIN_IND = 0;
static const unsigned int S_ENFORCEMENT_CLAUSE_MAX_IND = 7;



/** 
 * ------------------------------------------------------------------
 * Contract clause enforcement frequency.
 *
 * WARNING:  EnforcementFrequencyEnum and S_ENFORCEMENT_FREQUENCY 
 *   _must_ be kept in sync.  That is, any changes to one must 
 *   correspond to changes in the other.
 * ------------------------------------------------------------------
 */
typedef enum EnforcementFrequency__enum {
  EnforcementFrequency_NEVER           = 0,
  EnforcementFrequency_ALWAYS          = 1,
  EnforcementFrequency_ADAPTIVE_FIT    = 2,
  EnforcementFrequency_ADAPTIVE_TIMING = 3,
  EnforcementFrequency_PERIODIC        = 4,
  EnforcementFrequency_RANDOM          = 5
} EnforcementFrequencyEnum;
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MIN 
                                      = EnforcementFrequency_NEVER;
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MAX
                                      = EnforcementFrequency_RANDOM;

/**
 * Names corresponding to (and indexed by) EnforcementFrequencyEnum.
 */
static const char* S_ENFORCEMENT_FREQUENCY[6] = {
  "Never",
  "Always",
  "AdaptiveFit",
  "AdaptiveTiming",
  "Periodic",
  "Random"
};
static const unsigned int S_ENFORCEMENT_FREQUENCY_MIN_IND = 0;
static const unsigned int S_ENFORCEMENT_FREQUENCY_MAX_IND = 5;

#ifdef __cplusplus
}
#endif

#endif /* contractOptions_h */
