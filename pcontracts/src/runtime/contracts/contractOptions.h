/**
 * File:           contractOptions.h
 * Author:         T. Dahlgren
 * Created:        2012 April 12
 * Last Modified:  2012 August 17
 * 
 * @file
 * @section DESCRIPTION
 * Interface contract enforcement options.  The options are borrowed heavily 
 * from Babel's SIDL.
 *
 * @section LICENSE
 * TBD
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
 * Contract clause types. 
 *
 * WARNING:  ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
 *   MUST be kept in sync.  That is, any changes to one must correspond to 
 *   changes in the others.
 */
typedef enum ContractClause__enum {
  /** No contract clause. */
  ContractClause_NONE          = 0,
  /** Invariant contract clause. */
  ContractClause_INVARIANT     = 1,
  /** Precondition contract clause. */
  ContractClause_PRECONDITION  = 2,
  /** Postcondition contract clause. */
  ContractClause_POSTCONDITION = 4
} ContractClauseEnum;

/**
 * Names corresponding to (and indexable by) ContractClauseEnum.
 *
 * WARNING:  ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
 *   MUST be kept in sync.  That is, any changes to one must correspond to 
 *   changes in the others.
 */
static const char* S_CONTRACT_CLAUSE[5] = {
  "None",
  "Invariant",
  "Precondition",
  "**undefined**",
  "Postcondition"
};

/**
 * The minimum Contract Clause name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_CONTRACT_CLAUSE_MIN_IND = 0;

/**
 * The maximum Contract Clause name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_CONTRACT_CLAUSE_MAX_IND = 4;


/**
 * Enforcement clause options.
 *
 * WARNING:  ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
 *   MUST be kept in sync.  That is, any changes to one must correspond to 
 *   changes in the others.
 */
typedef enum EnforcementClause__enum {
  /** No contract clauses. */
  EnforcementClause_NONE           = ContractClause_NONE, /* 0 */
  /** Invariant clauses ONLY. */
  EnforcementClause_INVARIANTS     = ContractClause_INVARIANT, /* 1 */
  /** Precondition clauses ONLY. */
  EnforcementClause_PRECONDITIONS  = ContractClause_PRECONDITION, /* 2 */
  /** Invariant and Precondition clauses ONLY. */
  EnforcementClause_INVPRE         = ContractClause_INVARIANT 
                                   | ContractClause_PRECONDITION, /* 3 */
  /** Postcondition clauses ONLY. */
  EnforcementClause_POSTCONDITIONS = ContractClause_POSTCONDITION, /* 4 */
  /** Invariant and Postcondition clauses ONLY. */
  EnforcementClause_INVPOST        = ContractClause_INVARIANT 
                                   | ContractClause_POSTCONDITION, /* 5 */
  /** Precondition and Postcondition clauses ONLY. */
  EnforcementClause_PREPOST        = ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION, /* 6 */
  /** All contract clauses. */
  EnforcementClause_ALL            = ContractClause_INVARIANT
                                   | ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION /* 7 */
} EnforcementClauseEnum;

/**
 * The minimum Enforcement Clause enumeration value.  Provided for
 * traversal purposes.
 */
static const EnforcementClauseEnum S_ENFORCEMENT_CLAUSE_MIN 
                                   = EnforcementClause_NONE;
/**
 * The maximum Enforcement Clause enumeration value.  Provided for
 * traversal purposes.
 */
static const EnforcementClauseEnum S_ENFORCEMENT_CLAUSE_MAX
                                   = EnforcementClause_ALL;

/**
 * Names corresponding to (and indexable by) EnforcementClauseEnum.
 *
 * NOTE:  While the names shown below could be derived from the
 *   actual clauses at runtime, it was decided to maintain them
 *   here for consistency with the other enforcement-related
 *   enumerations.
 */
static const char* S_ENFORCEMENT_CLAUSE[8] = {
  "None",
  "Invariants",
  "Preconditions",
  "Invariants-Preconditions",
  "Postconditions",
  "Invariants-Postconditions",
  "Preconditions-Postconditions",
  "Invariants-Preconditions-Postconditions"
};

/**
 * Abbreviated names corresponding to (and indexable by) EnforcementClauseEnum.
 *
 * NOTE:  While the names shown below could be derived from the
 *   actual clauses at runtime, it was decided to maintain them
 *   here for consistency with the other enforcement-related
 *   enumerations.
 */
static const char* S_ENFORCEMENT_CLAUSE_ABBREV[8] = {
  "None",
  "Inv",
  "Pre",
  "InvPre",
  "Post",
  "InvPost",
  "PrePost",
  "InvPrePost"
};

/**
 * The minimum Enforcement Clause name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_ENFORCEMENT_CLAUSE_MIN_IND = 0;

/**
 * The maximum Enforcement Clause name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_ENFORCEMENT_CLAUSE_MAX_IND = 7;



/** 
 * Contract clause enforcement frequency options.
 *
 * WARNING:  EnforcementFrequencyEnum and S_ENFORCEMENT_FREQUENCY 
 *   MUST be kept in sync.  That is, any changes to one must 
 *   correspond to changes in the other.
 */
typedef enum EnforcementFrequency__enum {
  /** Never check contract clauses. */
  EnforcementFrequency_NEVER           = 0,
  /** Always check contract clauses. */
  EnforcementFrequency_ALWAYS          = 1,
  /** Adaptively check contract clauses based on local relative cost. */
  EnforcementFrequency_ADAPTIVE_FIT    = 2,
  /** Adaptively check contract clauses based on global relative cost. */
  EnforcementFrequency_ADAPTIVE_TIMING = 3,
  /** Periodically check contract clauses. */
  EnforcementFrequency_PERIODIC        = 4,
  /** Randomly check contract clauses. */
  EnforcementFrequency_RANDOM          = 5
} EnforcementFrequencyEnum;

/**
 * The minimum Enforcement Frequency enumeration value.  Provided for
 * traversal purposes.
 */
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MIN 
                                      = EnforcementFrequency_NEVER;

/**
 * The maximum Enforcement Frequency enumeration value.  Provided for
 * traversal purposes.
 */
static const EnforcementFrequencyEnum S_ENFORCEMENT_FREQUENCY_MAX
                                      = EnforcementFrequency_RANDOM;

/**
 * Names corresponding to (and indexable by) EnforcementFrequencyEnum.
 *
 * WARNING:  EnforcementFrequencyEnum and S_ENFORCEMENT_FREQUENCY 
 *   MUST be kept in sync.  That is, any changes to one must 
 *   correspond to changes in the other.
 */
static const char* S_ENFORCEMENT_FREQUENCY[6] = {
  "Never",
  "Always",
  "AdaptiveFit",
  "AdaptiveTiming",
  "Periodic",
  "Random"
};

/**
 * The minimum Enforcement Frequency name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_ENFORCEMENT_FREQUENCY_MIN_IND = 0;

/**
 * The maximum Enforcement Frequency name index.  Provided for traversal 
 * purposes.
 */
static const unsigned int S_ENFORCEMENT_FREQUENCY_MAX_IND = 5;

#ifdef __cplusplus
}
#endif

#endif /* contractOptions_h */
