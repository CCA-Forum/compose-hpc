#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "rose.h"
#include "Annotation.h"

using namespace std;

class Transform {
protected:
  SgProject *root;
public:
  Transform(SgProject *);
  static Transform *get_transform(SgProject *, Annotation *);
  virtual void generate() = 0;
};

class AbsorbStructTransform : public Transform {
  string struct_name;
public:
  AbsorbStructTransform(const string s, SgProject *root);
  virtual void generate();
};

#endif
