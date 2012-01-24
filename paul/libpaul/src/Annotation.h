#ifndef __ANNOTATION_H__
#define __ANNOTATION_H__

#include <string>
#include "rose.h"

using namespace std;

/*!
 AnnotationValue is a polymorphic class for subclassing the 'sum type' for each
 of the types of annotations.
*/
class AnnotationValue {
  virtual void merge(AnnotationValue *other) {}
};

/**
 * The annotation class represents a generic annotation to be attached by
 * PAUL to the AST nodes.  All annotations will provide a pointer back to
 * the SgNode to which they were attached and the original unparsed
 * comment string.  It is expected that specific types of annotations will
 * inherit from this class and provide additional methods that allow access
 * to the specific kind of information that the annotation type should
 * provide.
 */
class Annotation : public AstAttribute {
 protected:
  string  originalValue;
  SgLocatedNode *originalNode;
  string  tagName;
  AnnotationValue *sumtype;

 public:

  Annotation
    (const string s, SgLocatedNode *n, const string tag, AnnotationValue *v)
  {
    originalValue = s;
    originalNode = n;
    tagName = tag;
    sumtype = v;
  }

  AnnotationValue *getValue() {
    return sumtype;
  }

  string getValueString() {
    return originalValue;
  }

  const string getTag() {
    return tagName;
  }

  const SgLocatedNode *associatedElement() {
    return originalNode;
  }

};

#endif // __ANNOTATION_H__
