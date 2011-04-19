#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "rose.h"
#include "Annotation.h"

using namespace std;

class Transform {
protected:
  SgLocatedNode *root;
public:
  Transform(SgLocatedNode *);
  static Transform *get_transform(SgLocatedNode *, Annotation *);
  virtual void generate() = 0;
};

class AbsorbStructTransform : public Transform {
  string struct_name;
public:
  AbsorbStructTransform(const string s, SgLocatedNode *root);
  virtual void generate();
};

#endif
