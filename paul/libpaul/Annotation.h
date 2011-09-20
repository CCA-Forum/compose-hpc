#ifndef __ANNOTATION_H__
#define __ANNOTATION_H__

#include <string>
#include "rose.h"

using namespace std;

/**
 * The annotation class represents a generic annotation to be attached by
 * PAUL to the AST nodes.  All annotations will provide a pointer back to
 * the SgNode to which they were attached and the original unparsed
 * comment string.  It is expected that specific types of annotations will
 * inherit from this class and provide additional methods that allow access
 * to the specific kind of information that the annotation type should
 * provide.
 */
class Annotation {
 protected:
  string  originalValue;
  SgNode *originalNode;

 public:
  Annotation(string s, SgNode *n) {
    std::cout << "ANNOTATION CTR: " << s << std::endl;

    originalValue = s;
    originalNode = n;
  }

  const SgNode *associatedElement() {
    return originalNode;
  }

  string        textValue() {
    return originalValue;
  }
};

#endif // __ANNOTATION_H__
