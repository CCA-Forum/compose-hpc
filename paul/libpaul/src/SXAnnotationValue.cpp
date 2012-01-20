#include "SXAnnotationValue.h"
#include <sstream>

SXAnnotationValue::SXAnnotationValue(string s)
{
  sx = SExpr::parse(s);
}

void SXAnnotationValue::print() {
  cout << sx->toString() << endl;
}

SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p) {
  return dynamic_cast<SXAnnotationValue *>(p);
}

