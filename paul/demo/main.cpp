#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"

using namespace std;

// Bad, I know...
SgProject *root;

int main(int argc, char **argv) {
  string filename = argv[1];
  string exfile = filename;

  size_t fn = filename.find_last_of("/");
  if(fn != string::npos) exfile = filename.substr(fn+1,filename.length());

  fn = exfile.find_last_of(".");
  if(fn != string::npos) exfile = exfile.substr(0,fn);

  //string sdf = argv[2];
  //cout << sdf << endl;

  // create new file only for 1st time, overwrite if exists.
  int fileCount=0;

  root = frontend(argc,argv);

  //generateDOT(*root);

  CommentVisitor v(exfile, &fileCount);

  v.traverseInputFiles(root,preorder);

  return 0;
}
