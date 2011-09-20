#include "KVAnnotation.h"

KVAnnotation::KVAnnotation(string s, SgNode *n) : Annotation::Annotation(s,n) {
  std::cout << "KVANNOTATION CTR: " << s << std::endl;
  // call KV parser on s, ignore n
}

const string *KVAnnotation::lookup(string key) {
  key_value_map::iterator it;

  it = kvmap.find(key);
  if (it == kvmap.end()) {
    return NULL;
  } else {
    return &it->second;
  }
}
