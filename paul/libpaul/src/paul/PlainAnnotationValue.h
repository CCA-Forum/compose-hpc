#ifndef __PLAINANNOTATIONVALUE_H__
#define __PLAINANNOTATIONVALUE_H__

#include "Annotation.h"

class PlainAnnotationValue : public AnnotationValue {
 protected:
  string value;

 public:
  PlainAnnotationValue(string s);

  string getValue();

  void print();

  friend PlainAnnotationValue* isPlainAnnotationValue(AnnotationValue *p);

  void merge(PlainAnnotationValue *other);
};

PlainAnnotationValue* isPlainAnnotationValue(AnnotationValue *p);

#endif // __PLAINANNOTATIONVALUE_H__
