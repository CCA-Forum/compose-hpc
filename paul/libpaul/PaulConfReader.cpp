#include <iostream>
#include <fstream>
#include <map>
#include "PaulConfReader.h"

using namespace std;

const int  max_line_length = 512;
const bool verbose         = true;

void log(string s) {
  if (verbose) {
    cerr << s << endl;
  }
}

bool isEOL(char c) {
  if (c == '\r' || c == '\n') return true;
  return false;
}

bool isWhitespace(char c) {
  if (c == ' ' || c == '\t' || c == '\f' || c == '\v') return true;
  return false;
}

string strip_lead_trail_whitespace(string s) {
  string snew = s;
  size_t found;
  string whitespaces (" \t\f\v");

  // find leading
  found = snew.find_first_not_of(whitespaces);  
  if (found != string::npos) {
    snew = snew.substr(found,string::npos);
  } else {
    snew.clear();
    return snew;
  }

  // find trailing
  found = snew.find_last_not_of(whitespaces);
  if (found != string::npos) {
    snew.erase(found+1);
  } else {
    snew.clear(); // all whitespace
  }

  // return cleaned string
  return snew;
}

paul_tag_map read_paul_conf(string fname) {
  ifstream     conf_file;
  paul_tag_map tagmap;
  char         line[max_line_length]; // hard upper limit on line length.

  conf_file.open(fname.c_str(), ifstream::in);

  while (conf_file.good()) {
    conf_file.getline(line, max_line_length);

    if (line[0] == ';') {
      // comment line
      continue;
    }

    string tag, value;
    bool onTag = true;

    // process line.  tag must be a single word.  value can be any
    // string (potentially containin whitespace) until the end of the line.
    // note that all leading and trailing whitespace is removed from
    // both tag and value before finishing.
    for (int i = 0; line[i] != '\0' && i < max_line_length; i++) {
      if (isEOL(line[i]) && onTag) {
        break;
      }
      if (isWhitespace(line[i]) && onTag) {
        onTag = false;
        continue;
      }

      if (onTag)
        tag += line[i];
      else
        value += line[i];
    }

    value = strip_lead_trail_whitespace(value);
    tag = strip_lead_trail_whitespace(tag);

    if (tag.length() > 0) {
      tagmap[tag] = value;
    }
  }
  conf_file.close();

  return tagmap;
}

int main(int argc, char **argv) {
  paul_tag_map ptm;

  ptm = read_paul_conf("example.paulconf");

  paul_tag_map::iterator it;

  for (it = ptm.begin(); it != ptm.end(); it++) {
    cout << (*it).first << " => " << (*it).second << endl;
  }

  return 0;
}
