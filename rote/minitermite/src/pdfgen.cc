/*
Copyright 2006 Christoph Bonitz (christoph.bonitz@gmail.com)
see LICENSE in the root folder of this project
*/
/*
 * generatePrologTerm.C
 * Purpose: create a PROLOG representation of a given AST
 * Author: Christoph Bonitz (after looking at the documentedExamples) 
 */


#include <iostream>
#include <satire_rose.h>

int main ( int argc, char ** argv ) {
	//frontend processing
	SgProject * root = frontend(argc,argv);
	

	//Create dot and pdf files
	//DOT generation (numbering:preoder)
	AstDOTGeneration dotgen;
	dotgen.generateInputFiles(root,AstDOTGeneration::PREORDER);
	//PDF generation
	AstPDFGeneration pdfgen;
	pdfgen.generateInputFiles(root);
	std::cout << root->unparseToCompleteString();
	//create prolog term
	return 0;

       
}
