#ifndef __SXANNOTATIONVALUE_H__
#define __SXANNOTATIONVALUE_H__

#include "Annotation.h"
#include "sexp.h"

using namespace std;

/**
 * The SXAnnotation class is a specialization of the generic annotation
 * class for PAUL to represent symbolic expressions.  The s-expression
 * data structure is provided by the s-expression parsing library.
 *
 * TODO: Replace the C-based sexpr_t representation with a more C++-ish
 *       object that provides expected things like iterators and uses
 *       stl strings or a BOOST-based string under the covers instead of
 *       the cstring code currently in there.
 */
class SXAnnotationValue : public AnnotationValue {
 protected:
  sexp_t *sx;

 public:
  SXAnnotationValue(string s);

  sexp_t *getExpression()
  {
   return sx;
  }

  void print ();

  friend SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p);

};

SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p);

#endif // __SXANNOTATIONVALUE_H__
