#include "Transforms.h"

using namespace SageInterface;

Transform::Transform(SgLocatedNode *theroot) {
  root = theroot;
}

Transform *Transform::get_transform(SgLocatedNode *theroot,Annotation *ann) {
  if(ann->get_id() == "ABSORB_STRUCT_ARRAY") {
    return new AbsorbStructTransform(ann,theroot);
  }
  else {
    cerr << "Unknown annotation: " << ann->get_id() << endl;
    exit(1);
  }
}

AbsorbStructTransform::AbsorbStructTransform(Annotation *a,SgLocatedNode *p) 
: Transform(p) {
  string allocStr = a->get_attrib("outerAllocMethod")->string_value();
  if(allocStr == "stack") {
    // ok
    return;
  }
  if(allocStr == "dynamic") {
    cerr << "Dynamic allocation is not currently supported" << endl;
    exit(1);
  }
  else {
    cerr << "Allocation method not recognized: " << allocStr << endl;
    exit(1);
  }
  
}

void AbsorbStructTransform::generate() {
  SgClassDeclaration *clsDecl = isSgClassDeclaration(root);
  cerr << "Generating ABSORB_STRUCT_ARRAY for struct "
       << clsDecl->get_mangled_name().str() 
       << endl;
  if(!clsDecl) {
    cerr << "ABSORB_STRUCT_ARRAY must be attached to a struct, found"
         << root->class_name() 
         << endl;
    exit(1);
  }
  SgClassDefinition *def = clsDecl->get_definition();
  int n = def->get_members().size();
  
  cout << "@def@"                                      << endl;
  cout << "identifier s;"                              << endl;
  for(int i=1; i <= n; i++) {
    cout << "identifier x" << i << ";"                 << endl;
    cout << "type T" << i << ";"                       << endl;
  }
  cout << "@@"                                         << endl;
  cout << "struct s {"                                 << endl;
  for(int i=1; i <= n; i++) {
    cout << "- T" << i << " x" << i << ";"             << endl;
    cout << "+ T" << i << " *x" << i << ";"            << endl;
  }
  cout << "};"                                         << endl;
  cout                                                 << endl;
  cout << "@decl@"                                     << endl;
  cout << "identifier def.s,k;"                        << endl;
  cout << "@@"                                         << endl;
  cout << "- struct s *k;"                             << endl;
  cout << "+ struct s k;"                              << endl;
  cout                                                 << endl;
  cout << "@@"                                         << endl;
  cout << "function foo;"                              << endl;
  cout << "identifier def.s,k,x;"                      << endl;
  cout << "expression E1;"                             << endl;
  cout << "@@"                                         << endl;
  cout << "foo(...,"                                   << endl;
  cout << "- struct s *k"                              << endl;
  cout << "+ struct s k"                               << endl;
  cout << ",...) {"                                    << endl;
  cout << "<..."                                       << endl;
  cout << "- k[E1].x"                                  << endl;
  cout << "+ k.x[E1]"                                  << endl;
  cout << "...>"                                       << endl;
  cout << "}"                                          << endl;
  cout                                                 << endl;
  cout << "@@"                                         << endl;
  cout << "identifier decl.k,def.s;"                   << endl;
  for(int i=1; i <= n; i++) {
    cout << "identifier def.x" << i << ";"             << endl;
    cout << "type def.T" << i << ";"                   << endl;
  }
  cout << "expression E;"                              << endl;
  cout << "@@"                                         << endl;
  cout << "- k = malloc(E * sizeof(struct s));"        << endl;
  for(int i=1; i <= n; i++) {
    cout << "+ k.x" << i << " = malloc(E * sizeof(T" << i << "));" << endl;
  }
  cout << "..."                                        << endl;
  cout << "- free(k);"                                 << endl;
  for(int i=1; i <= n; i++) {
    cout << "+ free(k.x" << i << ");"                  << endl;
  }
}
