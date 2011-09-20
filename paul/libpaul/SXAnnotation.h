#ifndef __SXANNOTATION_H__
#define __SXANNOTATION_H__

#include "Annotation.h"
#include "sexp.h"

typedef struct sexpr sexpr_t;

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
class SXAnnotation : public Annotation {
 protected:
  sexpr_t *sx;

 public:
  SXAnnotation(string s, SgNode *n);
  sexpr_t *getExpression();
};

#endif // __SXANNOTATION_H__
