#include "PlainAnnotationValue.h"

PlainAnnotationValue::PlainAnnotationValue(string s) {
  value = s;
}

string PlainAnnotationValue::getValue() {
  return value;
}

void PlainAnnotationValue::print() {
  cerr << value << endl;
}

PlainAnnotationValue* isPlainAnnotationValue(AnnotationValue *p) {
  return dynamic_cast<PlainAnnotationValue *>(p);
}

void PlainAnnotationValue::merge(PlainAnnotationValue *other) {
  value = value + other->getValue();
}
