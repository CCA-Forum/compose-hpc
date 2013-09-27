/**
 * \internal
 * File:           contractClauseTypes.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 September 27
 * \endinternal
 *
 * @file
 * @brief
 * Extra contract clause enumerations and supporting types.
 *
 * @htmlinclude copyright.html
 */

#ifndef include_contract_Clause_Types_hpp
#define include_contract_Clause_Types_hpp

#include <string>
#include "contractOptions.h"


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
static const std::string ReservedWordPairs[][2] = {
  { "pce_all", "PCE_ALL" },
  { " and ", " && " },
  { "pce_any", "PCE_ANY" },
  { "pce_count", "PCE_COUNT" },
  { "pce_dimen", "PCE_DIMEN" },
  { " false ", " 0 " },
  { " iff ", "" },
  { "implies ", "" },
  { "pce_irange", "PCE_IRANGE" },
  { " is ", "" },
  { "pce_lower", "PCE_LOWER" },
  { "pce_max", "PCE_MAX" },
  { "pce_min", "PCE_MIN" },
  { " mod ", "" },
  { "pce_near_equal", "PCE_NEAR_EQUAL" },
  { "pce_non_decr", "PCE_NON_DECR" },
  { "pce_none", "PCE_NONE" },
  { "pce_non_incr", "PCE_NON_INCR" },
  { " not ", " !" },
  { " or ", " || " },
  { " pure", "" },
  { "pce_range", "PCE_RANGE" },
  { " rem ", "" },
  { "pce_result", "pce_result" },
  { "pce_size", "PCE_SIZE" },
  { "pce_stride", "PCE_STRIDE" },
  { "pce_sum", "PCE_SUM" },
  { " true ", " 1 " },
  { "pce_upper", "PCE_UPPER" },
  { " xor ", "" },
};
static const int MIN_NEE_INDEX = 0;
static const int MAX_NEE_INDEX = 29;

static const std::string L_UNSUPPORTED_EXPRESSION 
    = "Unsupported reserved word(s) detected in:\n\t";

#endif /* include_contract_Clause_Types_hpp */
