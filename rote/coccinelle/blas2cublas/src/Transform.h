#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "rose.h"
#include "KVAnnotationValue.h"
#include <string.h>

using namespace std;

class BlasToCublasTransform {
protected:
    SgNode *root;

public:

    // prefix for variables introduced
    // as part of the transformation.
    string arrayPrefix;

    // Check if prefix, vector lengths are provided.
    BlasToCublasTransform(KVAnnotationValue*, SgNode*);
    // Generate appropriate Coccinelle rules.
    void generate(string, int*, int*);
    ~BlasToCublasTransform() { }
};

#endif
