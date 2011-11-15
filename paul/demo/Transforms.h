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
  virtual void generate(string, int*) = 0;
};

class AbsorbStructTransform : public Transform {
public:
  AbsorbStructTransform(Annotation *a, SgLocatedNode *root);
  virtual void generate(string, int*);
};

class TascelTransform : public Transform {
public:
  int version;
  TascelTransform(Annotation *a, SgLocatedNode *root);
  virtual void generate(string, int*);
};

class BlasToCublasTransform : public Transform {
public:
  // prefix for variables introduced
  // as part of the transformation.
  string arrayPrefix;

  // Check if prefix, vector lengths are provided.
  BlasToCublasTransform(Annotation *a, SgLocatedNode *root);
  // Generate appropriate Coccinelle rules.
  virtual void generate(string, int*);
};

#endif
