/**
 * \internal
 * File:           contractClauseTypes.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2013 October 11
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
 * Non-executable assertion expression reserved words.
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
static const std::string ReservedWords[] = {
  "is initialization",
  "pce_all", 
  "pce_any",
  "pce_count",
  "pce_dimen",
  //"pce_in_range",
  "pce_lower",
  "pce_max",
  "pce_min",
  //"pce_near_equal",
  "pce_non_decr",
  "pce_none",
  "pce_non_incr",
  "pce_range",
  "pce_result",
  "pce_size",
  "pce_stride",
  "pce_sum",
  "pce_upper",
  "is pure",
};
static const int MIN_NEE_INDEX = 0;
static const int MAX_NEE_INDEX = 29;

static const std::string L_UNSUPPORTED_EXPRESSION 
    = "Unsupported reserved word(s) detected in:\n\t";

#endif /* include_contract_Clause_Types_hpp */
