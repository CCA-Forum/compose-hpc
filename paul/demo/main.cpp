#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"

using namespace std;

// Bad, I know...
SgProject *root;

int main(int argc, char **argv) {
  root = frontend(argc,argv);
  CommentVisitor v;
  v.traverseInputFiles(root,preorder);
  return 0;
}
