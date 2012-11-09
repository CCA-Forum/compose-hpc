/**
 * File:           contractClauseTypes.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2012 November 1
 *
 * @file
 * @section DESCRIPTION
 * Basic contract clause-related type definitions and enumerations.
 *
 * @section SOURCE
 * This code was originally part of the initial ContractInstrumenter.cpp,
 * which was renamed to RoutineContractInstrumenter.cpp.
 *
 * @section LICENSE
 * TBD
 */

#ifndef include_contract_Clause_Types_hpp
#define include_contract_Clause_Types_hpp

#include <string>
#include "contractOptions.h"

using namespace std;


/**
 * Assertion expression support states.
 */
typedef enum AssertionSupport__enum {
  /** ADVISORY:  Advisory only. */
  AssertionSupport_ADVISORY,
  /** EXECUTABLE:  Executable in C (currently as-is). */
  AssertionSupport_EXECUTABLE,
  /** UNSUPPORTED:  Known to include an unsupported annotation. */
  AssertionSupport_UNSUPPORTED
} AssertionSupportEnum;


/**
 * Supported structured contract comment types.
 */
typedef enum ContractComment__enum {
  /** NONE:  No contract comment present. */
  ContractComment_NONE,
  /** INVARIANT:  An invariant clause comment. */
  ContractComment_INVARIANT,
  /** PRECONDITION:  A precondition clause comment. */
  ContractComment_PRECONDITION,
  /** POSTCONDITION:  A postcondition clause comment. */
  ContractComment_POSTCONDITION,
  /** INIT:  An initialization clause comment. */
  ContractComment_INIT,
  /** FINAL:  A finalization clause comment. */
  ContractComment_FINAL
} ContractCommentEnum;


/**
 * Mapping of contract comment to contract clauses.  MUST be kept in sync
 * with ContractCommentEnum and reflect corresponding ContractClauseEnum
 * entries.
 */
static const ContractClauseEnum ContractCommentClause[] = {
  /** NONE:  No corresponding contract clause. */
  ContractClause_NONE,
  /** INVARIANT:  Invariant contract clause. */
  ContractClause_INVARIANT,
  /** PRECONDITION:  Precondition contract clause. */
  ContractClause_PRECONDITION,
  /** POSTCONDITION:  Postcondition contract clause. */
  ContractClause_POSTCONDITION,
  /** INIT:  No corresponding contract clause. */
  ContractClause_NONE,
  /** FINAL:  No corresponding contract clause. */
  ContractClause_NONE
};


/**
 * Non-executable assertion expression reserved words (provided for SIDL 
 * specifications).
 *
 * @todo Once expressions are parsed, this should be changed to a
 *    typdef map<string, string> dictType;
 *    typdef pair<string, string> dictEntryType;
 * then populate with:
 *   dictType ReservedWordDict;
 *   dictEntryType ReservedEntry;
 *   For each entry:
 *     ReservedWordDict.insert(ReservedEntry(<key>, <value>));
 * and access:
 *   string value = ReservedWordDict[<key>];
 *   if (value != "") {  // TBD: Distinguish keys with no values from non-keys
 *     // Use it
 *   }
 */
static const string ReservedWordPairs[][2] = {
  { "all", "PCE_ALL" },
  { "and", "&&" },
  { "any", "PCE_ANY" },
  { "count", "PCE_COUNT" },
  { "dimen", "PCE_DIMEN" },
  { "false", "0" },
  { "iff", "" },
  { "implies", "" },
  { "irange", "PCE_IRANGE" },
  { "is", "" },
  { "lower", "PCE_LOWER" },
  { "max", "PCE_MAX" },
  { "min", "PCE_MIN" },
  { "mod", "" },
  { "nearEqual", "PCE_NEAR_EQUAL" },
  { "nonDecr", "PCE_NON_DECR" },
  { "none", "PCE_NONE" },
  { "nonIncr", "PCE_NON_INCR" },
  { "not", "!" },
  { "or", "||" },
  { "pure", "" },
  { "range", "PCE_RANGE" },
  { "rem", "" },
  { "result", "pce_result" },
  { "size", "PCE_SIZE" },
  { "stride", "PCE_STRIDE" },
  { "sum", "PCE_SUM" },
  { "true", "1" },
  { "upper", "PCE_UPPER" },
  { "xor", "" },
};
static const int MIN_NEE_INDEX = 0;
static const int MAX_NEE_INDEX = 29;

static const string L_UNSUPPORTED_EXPRESSION 
    = "Unsupported reserved word(s) detected in:\n\t";

#endif /* include_contract_Clause_Types_hpp */
