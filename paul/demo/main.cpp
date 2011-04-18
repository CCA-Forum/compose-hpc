#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"

int main(int argc, char **argv) {
  SgProject *node = frontend(argc,argv);
  CommentVisitor v;
  v.traverseInputFiles(node,preorder);
  return 0;
}
