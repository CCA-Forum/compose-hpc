#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include "rose.h"
#include "Annotation.h"
/* key-value annotation class */
#include "KVAnnotationValue.h"
#include <string.h>

using namespace std;

class Transform {
protected:
	SgNode *root;
public:
  Transform(SgNode *);
  virtual void generate(string) = 0;
};

class CCSDTransform : public Transform {
public:
  string version;
  CCSDTransform(KVAnnotationValue *a, SgNode *root);
  void generate(string);
};

#endif
