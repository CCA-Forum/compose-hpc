#include "Transforms.h"

using namespace SageInterface;

Transform::Transform(SgLocatedNode *theroot) {
  root = theroot;
}

Transform *Transform::get_transform(SgLocatedNode *theroot,Annotation *ann) {
  if(ann->get_id() == "ABSORB_STRUCT_ARRAY") {
    string name = ann->get_attrib("structId")->string_value();
    return new AbsorbStructTransform(name,theroot);
  }
  else {
    cerr << "Unknown annotation: " << ann->get_id() << endl;
    exit(1);
  }
}

AbsorbStructTransform::AbsorbStructTransform(const string s,SgLocatedNode *p) 
: Transform(p) {
  struct_name = s;
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
  cerr << def->get_members().size() << endl;
}
