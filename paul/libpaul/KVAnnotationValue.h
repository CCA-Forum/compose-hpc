#ifndef __KVANNOTATIONVALUE_H__
#define __KVANNOTATIONVALUE_H__

#include <string>
#include <map>

#include "Annotation.h"
#include "Dynamic.h"

using namespace std;

/**
 * The KVAnnotationValue class is a specialization of the generic
 * AnnotationValue class for PAUL to represent key-value pairs.  The
 * Kvannotationvalue class allows callers to look up values by key.  The
 * underlying data structure that implements the key-value pair mapping is not
 * exposed.
 */
class KVAnnotationValue : public AnnotationValue {
 public:

  KVAnnotationValue(string s);

  Dynamic *lookup(string key);

  void print ();

  friend KVAnnotationValue* isKVAnnotationValue( AnnotationValue *p);

 private:

  // FIXME: Duplicated From parser.y
  typedef string Key;
  typedef Dynamic *Value;
  typedef map<Key,Value> KeyValueMap;

  KeyValueMap *kvmap;
  KeyValueMap *parse(const string input);
};

KVAnnotationValue* isKVAnnotationValue( AnnotationValue *p);

#endif // __KVANNOTATIONVALUE_H__
