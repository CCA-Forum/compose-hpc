#ifndef __SXANNOTATIONVALUE_H__
#define __SXANNOTATIONVALUE_H__

#include "Annotation.h"
#include "ssexpr.h"

using namespace std;

/**
 * The SXAnnotation class is a specialization of the generic annotation
 * class for PAUL to represent symbolic expressions.  The s-expression
 * data structure is provided by the s-expression parsing library.
 */
class SXAnnotationValue : public AnnotationValue {
 protected:
  SExpr *sx;

 public:
  SXAnnotationValue(string s);

  SExpr *getExpression()
  {
   return sx;
  }

  void print ();

  friend SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p);

  void merge( SXAnnotationValue* other);

};

SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p);

#endif // __SXANNOTATIONVALUE_H__
