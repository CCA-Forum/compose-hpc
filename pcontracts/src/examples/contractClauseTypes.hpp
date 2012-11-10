/**
 * File:           contractClauseTypes.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2012 November 9
 *
 * @file
 * @section DESCRIPTION
 * Extra contract clause enumerations and types.
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
