/**
 * \internal
 * File:           contractOptions.h
 * Author:         T. Dahlgren
 * Created:        2012 April 12
 * Last Modified:  2013 March 12
 * \endinternal
 *
 * @file
 * @brief  
 * Basic interface contract enforcement options.
 *
 * @htmlinclude contractsSource.html
 * @htmlinclude copyright.html
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
 * \warning ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
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
  ContractClause_POSTCONDITION = 4,
  /** Assertion contract clause. */
  ContractClause_ASSERT = 8
} ContractClauseEnum;

/**
 * Names corresponding to (and indexable by) ContractClauseEnum.
 *
 * \warning ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
 *   MUST be kept in sync.  That is, any changes to one must correspond to 
 *   changes in the others.
 */
static const char* S_CONTRACT_CLAUSE[9] = {
  "None",
  "Invariant",
  "Precondition",
  "**undefined1**",
  "Postcondition",
  "**undefined2*",
  "**undefined3*",
  "**undefined4*",
  "Assertion"
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
static const unsigned int S_CONTRACT_CLAUSE_MAX_IND = 8;


/**
 * Enforcement clause options.
 *
 * \warning ContractClauseEnum, S_CONTRACT_CLAUSE, and EnforcementClauseEnum
 *   MUST be kept in sync.  That is, any changes to one must correspond to 
 *   changes in the others.
 */
typedef enum EnforcementClause__enum {
  /** Do not check any contract clauses. */
  EnforcementClause_NONE           = ContractClause_NONE, /* 0 */
  /** Check invariant clauses ONLY. */
  EnforcementClause_INVARIANTS     = ContractClause_INVARIANT, /* 1 */
  /** Check precondition clauses ONLY. */
  EnforcementClause_PRECONDITIONS  = ContractClause_PRECONDITION, /* 2 */
  /** Check invariant and precondition clauses ONLY. */
  EnforcementClause_INVPRE         = ContractClause_INVARIANT 
                                   | ContractClause_PRECONDITION, /* 3 */
  /** Check postcondition clauses ONLY. */
  EnforcementClause_POSTCONDITIONS = ContractClause_POSTCONDITION, /* 4 */
  /** Check invariant and postcondition clauses ONLY. */
  EnforcementClause_INVPOST        = ContractClause_INVARIANT 
                                   | ContractClause_POSTCONDITION, /* 5 */
  /** Check precondition and postcondition clauses ONLY. */
  EnforcementClause_PREPOST        = ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION, /* 6 */
  /** Check precondition, postcondition, and invariant contract clauses. */
  EnforcementClause_INVPREPOST     = ContractClause_INVARIANT
                                   | ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION, /* 7 */
  /** Check assert contract clauses ONLY. */
  EnforcementClause_ASSERTS        = ContractClause_ASSERT,        /* 8 */
  /** Check invariant and assert contract clauses. */
  EnforcementClause_INVASRT        = ContractClause_INVARIANT
                                   | ContractClause_ASSERT,        /* 9 */
  /** Check precondition and assert contract clauses. */
  EnforcementClause_PREASRT        = ContractClause_PRECONDITION
                                   | ContractClause_ASSERT,        /* 10 */
  /** Check invariant, precondition, and assert contract clauses. */
  EnforcementClause_INVPREASRT     = ContractClause_INVARIANT
                                   | ContractClause_PRECONDITION
                                   | ContractClause_ASSERT,        /* 11 */
  /** Check postcondition and assert contract clauses. */
  EnforcementClause_POSTASRT       = ContractClause_POSTCONDITION
                                   | ContractClause_ASSERT,        /* 12 */
  /** Check invariant, postcondition, and assert contract clauses. */
  EnforcementClause_INVPOSTASRT    = ContractClause_INVARIANT
                                   | ContractClause_POSTCONDITION
                                   | ContractClause_ASSERT,        /* 13 */
  /** Check precondition, postcondition, and assert contract clauses. */
  EnforcementClause_PREPOSTASRT    = ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION
                                   | ContractClause_ASSERT,        /* 14 */
  /** Check all contract clauses. */
  EnforcementClause_ALL            = ContractClause_INVARIANT
                                   | ContractClause_PRECONDITION
                                   | ContractClause_POSTCONDITION
                                   | ContractClause_ASSERT         /* 15 */
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
 * \note While these names could be derived from the actual clauses
 *  at runtime, it was decided to maintain them here for consistency 
 *  with other enforcement-related enumerations.
 */
static const char* S_ENFORCEMENT_CLAUSE[16] = {
  "None",
  "Invariants",
  "Preconditions",
  "Invariants-Preconditions",
  "Postconditions",
  "Invariants-Postconditions",
  "Preconditions-Postconditions",
  "Invariants-Preconditions-Postconditions",
  "Assertions",
  "Invariants-Assertions",
  "Preconditions-Assertions",
  "Invariants-Preconditions-Assertions",
  "Postconditions-Assertions",
  "Invariants-Postconditions-Assertions",
  "Preconditions-Postconditions-Assertions",
  "Invariants-Preconditions-Postconditions-Assertions"
};

/**
 * Abbreviated names corresponding to (and indexable by) EnforcementClauseEnum.
 *
 * \note While these names could be derived from the actual clauses
 *  at runtime, it was decided to maintain them here for consistency 
 *  with other enforcement-related enumerations.
 */
static const char* S_ENFORCEMENT_CLAUSE_ABBREV[16] = {
  "None",
  "Inv",
  "Pre",
  "InvPre",
  "Post",
  "InvPost",
  "PrePost",
  "InvPrePost",
  "Asrt",
  "InvAsrt",
  "PreAsrt",
  "InvPreAsrt",
  "PostAsrt",
  "InvPostAsrt",
  "PrePostAsrt",
  "InvPrePostAsrt"
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
static const unsigned int S_ENFORCEMENT_CLAUSE_MAX_IND = 15;


/** 
 * Contract clause enforcement frequency options.
 *
 * \warning EnforcementFrequencyEnum and S_ENFORCEMENT_FREQUENCY 
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
 * \warning EnforcementFrequencyEnum and S_ENFORCEMENT_FREQUENCY 
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
