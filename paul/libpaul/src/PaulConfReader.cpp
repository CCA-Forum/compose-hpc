#include <cctype>
#include <iostream>
#include <fstream>
#include <map>
#include "PaulConfReader.h"
#include "Utilities.h"

using namespace std;

const int  max_line_length = 512;
const bool verbose         = true;

void log(string s) {
  if (verbose) {
    cerr << s << endl;
  }
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
      if (is_eol(line[i]) && onTag) {
        break;
      }
      if (isspace(line[i]) && onTag) {
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

/**

int main(int argc, char **argv) {
  paul_tag_map ptm;

  ptm = read_paul_conf("example.paulconf");

  paul_tag_map::iterator it;

  for (it = ptm.begin(); it != ptm.end(); it++) {
    cout << (*it).first << " => " << (*it).second << endl;
  }

  return 0;
}

**/
