/**
 * \internal
 * File:           contractClauseTypes.hpp
 * Author:         T. Dahlgren
 * Created:        2012 November 1
 * Last Modified:  2015 February 5
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
 * Non-executable assertion expression interfaces.
 *
 * @todo Once expressions are parsed, this should be changed to a
 *    typdef map<string, string> dictType;
 *    typdef pair<string, string> dictEntryType;
 * then populate with:
 *   dictType UnsupportedDict;
 *   dictEntryType UnsupportedEntry;
 *   For each entry:
 *     UnsupportedDict.insert(UnsupportedEntry(<key>, <value>));
 * and access:
 *   string value = UnsupportedDict[<key>];
 *   if (value != "") {  // TBD: Distinguish keys with no values from non-keys
 *     // Use it
 *   }
 */

static const std::string UnsupportedInterfaces[] = {
  "is initialization",
  "pce_all", 
  //"pce_all_null",  /* Now supported as executable */
  "pce_any",
  //"pce_any_null",  /* Now supported as executable */
  "pce_count",
  "pce_dimen",
  //"pce_in_range", /* Now supported as executable */
  "pce_lower",
  //"pce_max",  /* Now supported as executable */
  //"pce_min",  /* Now supported as executable */
  //"pce_near_equal", /* Now supported as executable */
  "pce_non_decr",
  "pce_none",
  "pce_non_incr",
  //"pce_range",  /* Now supported as executable */
  //"pce_result", /* Now supported as executable */
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
