/**
 * File:           Utilities.h
 * Authors:        M. Sottile and T. Dahlgren
 * Created:        2012 October 11
 * Last Modified:  2012 October 11
 *
 * @file
 * @section DESCRIPTION
 * Basic, common utilities.
 *
 * @section SOURCE
 * Basic (non-ROSE) utilities taken from existing Paul and Paul Contracts
 * sources.
 *
 * @section LICENSE
 * TBD
 */

#ifndef include_Utilities_h
#define include_Utilities_h

#include <string.h>

using namespace std;


/**
 * Strip and compress white space.  Strip leading and trailing white space
 * and replace and compress embedded white space to at most one blank between
 * non-white space contents.
 *
 * @param txt  The text to be cleaned.
 * @return     The cleaned and compressed version of the string.
 */
string
compress(string txt);


/**
 * Determine if the specified character is EOL.
 *
 * @param c  The character being checked.
 * @return   True if the character is EOL; false otherwise.
 */
bool 
is_eol(char c);


/**
 * Remove leading white space from the string.
 *
 * @param txt  The text to be cleaned.
 * @return     The remainder of the string.
 */
string 
remove_leading_whitespace(const string txt);


/**
 * Remove leading and trailing white space.
 *
 * @param txt  The text to be cleaned.
 * @return     The remainder of the string.
 */
string 
strip_lead_trail_whitespace(string txt);

#endif /* include_Utilities_h */
