#include "Transforms.h"

using namespace SageInterface;

Transform::Transform(SgProject *theroot) {
  root = theroot;
}

Transform *Transform::get_transform(SgProject *theroot,Annotation *ann) {
  if(ann->get_id() == "ABSORB_STRUCT_ARRAY") {
    string name = ann->get_attrib("structId")->string_value();
    return new AbsorbStructTransform(name,theroot);
  }
  else {
    cerr << "Unknown annotation: " << ann->get_id() << endl;
    exit(1);
  }
}

AbsorbStructTransform::AbsorbStructTransform(const string s,SgProject *p) 
: Transform(p) {
  struct_name = s;
}

void AbsorbStructTransform::generate() {
  SgScopeStatement *scope = getFirstGlobalScope(root);
  SgType *t = lookupNamedTypeInParentScopes(struct_name,scope);
  SgClassType *typ = isSgClassType(t);
  if(typ) {
    SgDeclarationStatement *decl = typ->get_declaration();
    SgClassDeclaration *clsDecl = isSgClassDeclaration(decl);
    if(clsDecl) {
      const SgClassDefinition *def = clsDecl->get_definition();
      cout << def << endl;
      cout << typ->get_mangled().str() << endl;
    }
  }
}
