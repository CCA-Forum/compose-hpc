/**
 * File:           Utilities.cpp
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

#include <iostream>
#include <string.h>
#include "Utilities.h"

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
compress(string txt) {
  if (!txt.empty()) {
    unsigned int i;
    for (i=0; i<txt.length(); i++) {
      if (isspace(txt[i])) {
        txt[i] = ' ';
      }
    }

    size_t start = 0, end;
    while ( (end=txt.find("  ", start)) != string::npos ) {
      txt.replace(end, 2, 1, ' ');
      start = end;
    }

    start=txt.find_first_not_of(' ');
    end=txt.find_last_not_of(' ');
    if ( (start != string::npos) && (end != string::npos) ) {
      txt=txt.substr(start, end-start+1);
    }
  }

  return txt;
} /* compress */


/**
 * Determine if the specified character is EOL.
 *
 * @param c  The character being checked.
 * @return   True if the character is EOL; false otherwise.
 */
bool 
is_eol(char c) {
  if (c == '\r' || c == '\n') return true;
  return false;
}


/**
 * Remove leading blanks and tabs.
 *
 * @param txt  The text to be cleaned.
 */
string 
remove_leading_whitespace(const string txt) {
  unsigned int i = 0;
  while (i < txt.length() && isspace(txt[i])) {
    i++;
  }

  if (i == txt.length()) {
    return "";
  }

  return txt.substr(i);
}


/**
 * Remove leading and trailing white space.
 *
 * @param txt  The text to be cleaned.
 * @return     The remainder of the string.
 */
string 
strip_lead_trail_whitespace(string txt) {
  string newTxt = txt;
  size_t found;
  string whitespaces (" \t\f\v\r\n");

  // find leading
  found = newTxt.find_first_not_of(whitespaces);
  if (found != string::npos) {
    newTxt = newTxt.substr(found,string::npos);
  } else {
    newTxt.clear();
    return newTxt;
  }

  // find trailing
  found = newTxt.find_last_not_of(whitespaces);
  if (found != string::npos) {
    newTxt.erase(found+1);
  } else {
    newTxt.clear(); // all whitespace
  }

  // return cleaned string
  return newTxt;
}
