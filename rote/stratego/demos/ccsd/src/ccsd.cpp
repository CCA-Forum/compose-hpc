/* include rose */
#include "rose.h"

/* include paul decorate header */
#include "PaulDecorate.h"

#include "Transforms.h"

using namespace std;

class Visitor : public AstSimpleProcessing {
protected:
  string inputFile;
  void virtual visit(SgNode *node);
public:
   Visitor(string exfile) {
	  inputFile = exfile;
  }
};

//
// the visitor function
//
void Visitor::visit(SgNode *node) {
  //
  // if an annotation was attached, it would be called "CCSD" since
  // that was the tag we cared about.
  //
  Annotation *annot = (Annotation *)node->getAttribute("CCSD");

  // if this is null, no such annotation is attached, so return.
  if (annot == NULL) {
    //cout << "Annot is NULL" << endl;
    return;
  }

  KVAnnotationValue *val = (KVAnnotationValue *)annot->getValue();

  Transform *trans = new CCSDTransform(val,node);
  trans->generate(inputFile);

/*  // make sure it is a key-value annotation!
  val = isKVAnnotationValue(val);
  ROSE_ASSERT(val != NULL);
  val->print();
  cout << endl;*/
}

int main( int argc, char * argv[] )
{
	  //Get name of the input source file on which PAUL is run.
	  string filename = argv[1];
	  string exfile = filename;

	  size_t fn = exfile.find_last_of(".");
	  if(fn == string::npos) {
		  filename = argv[2];
		  exfile = filename;
	  }

	  // Extract input source file name from a relative/absolute path.
	  fn = filename.find_last_of("/");
	  if(fn != string::npos) exfile = filename.substr(fn+1,filename.length());

	  fn = exfile.find_last_of(".");
	  if(fn != string::npos) exfile = exfile.substr(0,fn);

	  //Build the AST used by ROSE
	  SgProject* sageProject  = frontend (argc , argv) ;
	  ROSE_ASSERT (sageProject != NULL);

	  // decorate the AST with the PAUL annotations
	  paulDecorate (sageProject, "ccsd.paulconf");

	  // Run internal consistency tests on AST
	  AstTests::runAllTests(sageProject);

	  // Generate DOT file to visualize the AST
	  generateDOT(*sageProject);

	  Visitor v(exfile);

	  v.traverseInputFiles(sageProject, preorder);

	  // Generate source code from AST and call the vendor's compiler
	  return backend(sageProject);
}
