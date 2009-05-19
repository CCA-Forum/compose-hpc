/* -*- C++ -*-
  Copyright 2009 Adrian Prantl <adrian@complang.tuwien.ac.at>
*/
#ifndef _AST_JANITOR_H
#define _AST_JANITOR_H
#include "satire_rose.h"
#include "TermToRose.h"

/**
 * Perform various tasks to generate a valid AST
 * including the setting of parent scopes an pointers
 */
#define FI Sg_File_Info::generateDefaultFileInfoForTransformationNode()

class InheritedAttribute
{
public:
  PrologToRose* ptr;
  SgScopeStatement* scope;
  SgNode* parent;
  // Specific constructors are required
  InheritedAttribute(PrologToRose* conv, 
		     SgScopeStatement* s = NULL, 
		     SgNode* p = NULL)
    : ptr(conv), scope(s), parent(p) 
  {};
  InheritedAttribute(const InheritedAttribute& X)
    : ptr(X.ptr), scope(X.scope), parent(X.parent) 
  {};
};

class AstJanitor : public AstTopDownProcessing<InheritedAttribute>
{
public:
  //virtual function must be defined
  virtual InheritedAttribute 
  evaluateInheritedAttribute(SgNode *n, InheritedAttribute attr)
  {
    // FileInfo
    if (SgLocatedNode* ln = isSgLocatedNode(n)) {
      // Set the CompilerGenerated Flag
      //Sg_File_Info* fi = ln->get_file_info();
      //fi->set_classificationBitField(fi->get_classificationBitField() 
      //				   | Sg_File_Info::e_compiler_generated 
      /*| Sg_File_Info::e_output_in_code_generation*/
      //);
      // Set EndOfConstruct
      ln->set_endOfConstruct(ln->get_startOfConstruct());
    }

    // Scope
    SgScopeStatement* scope = isSgScopeStatement(n);
    if (scope == NULL) scope = attr.scope;
    //    else {
      // rebuild the symbol table
      /// scope->set_symbol_table(NULL);
      //  SageInterface::rebuildSymbolTable(scope);
      //}

    if (SgDeclarationStatement* decl = isSgDeclarationStatement(n)) {
      // These nodes don't have a scope associated
      if (!isSgVariableDeclaration(decl) 
	  && !isSgFunctionParameterList(decl)
	  ) {
	ROSE_ASSERT(scope != NULL);
	decl->set_scope(scope);
      } 
      PrologToRose::addSymbol(scope, decl);
    }

    // Parent
    //ROSE_ASSERT(n->get_parent() == attr.parent);
    for (SgNode* n1 = n; 
	 isSgInitializedName(n) && 
	   (!isSgFunctionParameterList(   n->get_parent()) &&
	    !isSgVariableDeclaration(	  n->get_parent()) &&
	    !isSgCtorInitializerList(	  n->get_parent()) &&
	    !isSgProcedureHeaderStatement(n->get_parent())
	    ) ;  n1 = n1->get_parent()) {
      // Walk upwards until we reach a suitable node
      n->set_parent(n1->get_parent());
      SgGlobal* glob = isSgGlobal(n1->get_parent());
      if (glob || n1->get_parent() == NULL) {
	// Enum Decl inside of a typedef, for example
	SgInitializedName* iname = isSgInitializedName(n);
	SgVariableDeclaration* vdec = new SgVariableDeclaration(FI);
	vdec->append_variable(iname, iname->get_initializer());
	vdec->set_parent(glob);
	//vdec->setForward();
	vdec->set_definingDeclaration(vdec);
	n->set_parent(attr.ptr->createDummyNondefDecl(vdec, FI, "", 
					      iname->get_typeptr(), 
					      iname->get_initializer()));
      }
      //std::cerr<<"Setting parent of "<<n->class_name()
      //         <<" to "<<n->get_parent()->class_name()<<std::endl;
    } 

    if (isSgVariableDeclaration(n) && isSgForInitStatement(attr.parent))
      n->set_parent(attr.parent->get_parent());

    return InheritedAttribute(attr.ptr, scope, n);
  };
};


#endif
