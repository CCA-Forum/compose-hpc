#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "rose.h"
#include "KVAnnotationValue.h"
#include <string.h>

using namespace std;

class TascelTransform {
protected:
    SgNode *root;
public:
    int version;
    TascelTransform(KVAnnotationValue*, SgNode*);
    void generate(string, int*);
    ~TascelTransform() { }
};

#endif
