#include "SXAnnotationValue.h"
#include <sstream>

SXAnnotationValue::SXAnnotationValue(string s)
{
  sx = parse_sexp ((char *) s.c_str(), s.size());
}

void SXAnnotationValue::print() {

  cout << "length is " << sexp_list_length (sx) << endl;
}

SXAnnotationValue* isSXAnnotationValue( AnnotationValue *p) {
  return dynamic_cast<SXAnnotationValue *>(p);
}

