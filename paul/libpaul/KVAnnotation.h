#ifndef __KVANNOTATION_H__
#define __KVANNOTATION_H__

#include <string>
#include <map>

#include "Annotation.h"

using namespace std;

/**
 * The KVAnnotation class is a specialization of the generic annotation
 * class for PAUL to represent key-value pairs.  In addition to the
 * methods provided by the Annotation class, the KVAnnotation class allows
 * callers to look up values by key.  The underlying data structure that
 * implements the key-value pair mapping is not exposed.
 */
class KVAnnotation : public Annotation {
 public:
  typedef map<string, const string> key_value_map;

  KVAnnotation(string s, SgNode *n); 

  const string *lookup(string key);

 private:
  key_value_map kvmap;

};

#endif // __KVANNOTATION_H__
