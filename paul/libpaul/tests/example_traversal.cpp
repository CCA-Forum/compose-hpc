/* include rose */
#include "rose.h"

/* include paul decorate header */
#include "PaulDecorate.h"

/* key-value annotation class */
#include "KVAnnotationValue.h"

using namespace std;

//
// define a basic visitor class that we will use to demonstrate visiting
// nodes, looking to see if they have an annotation, and extracting and
// printing it if they do.
//
// For this example, we will assume that we are looking for annotations of
// tag "EXAMPLE" that are key-value pair typed.
//

class ExampleVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
public:
  ExampleVisitor() {}
};

//
// the visitor function
//
void ExampleVisitor::visit(SgNode *node) {
  //
  // if an annotation was attached, it would be called "EXAMPLE" since
  // that was the tag we cared about.
  //
  Annotation *annot = (Annotation *)node->getAttribute("EXAMPLE");

  // if this is null, no such annotation is attached, so return.
  if (annot == NULL) {
    return;
  }

  KVAnnotationValue *val = (KVAnnotationValue *)annot->getValue();

  // make sure it is a key-value annotation!
  val = isKVAnnotationValue(val);
  ROSE_ASSERT(val != NULL);

  cout << "Found annotated node:" << node->class_name() << endl;
  val->print();
  cout << endl;
}

//
// main
//
int main( int argc, char * argv[] )
{
  //Build the AST used by ROSE
  SgProject* sageProject  = frontend (argc , argv) ;
  ROSE_ASSERT (sageProject != NULL);

  // decorate the AST with the PAUL annotations
  paulDecorate (sageProject, "example.paulconf");
  
  // Run internal consistency tests on AST
  AstTests::runAllTests(sageProject);

  ExampleVisitor v;

  v.traverseInputFiles(sageProject, preorder);

  // Generate source code from AST and call the vendor's compiler
  //  return backend(sageProject);
}
