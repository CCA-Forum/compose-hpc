#include "SXAnnotationValue.h"
#include <sstream>

SXAnnotationValue::SXAnnotationValue(string s)
{
  sx = SExpr::parse(s);
}

void SXAnnotationValue::merge(SXAnnotationValue *other) {
  cerr << "Merge not supported for s-expressions yet." << endl;
}

void SXAnnotationValue::print() {
  cout << sx->toString() << endl;
}

SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p) {
  return dynamic_cast<SXAnnotationValue *>(p);
}

