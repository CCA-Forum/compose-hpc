#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"

using namespace std;

// Bad, I know...
SgProject *root;

int main(int argc, char **argv) {

  //Get name of the input source file on which PAUL is run.
  string filename = argv[1];
  string exfile = filename;

  // Extract input source file name from a relative/abosulte path.
  size_t fn = filename.find_last_of("/");
  if(fn != string::npos) exfile = filename.substr(fn+1,filename.length());

  fn = exfile.find_last_of(".");
  if(fn != string::npos) exfile = exfile.substr(0,fn);

  // variable to ensure creation of coccinelle rules file only for first time
  // a particular transformation is applied, overwrite if already exists from
  // previous runs on the same input source file.
  int fileCount=0;

  root = frontend(argc,argv);

  //generateDOT(*root);

  CommentVisitor v(exfile, &fileCount);

  v.traverseInputFiles(root,preorder);

  return 0;
}
