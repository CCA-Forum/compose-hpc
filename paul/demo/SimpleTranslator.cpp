//=========================================================================================
// Name        : SimpleTranslator.C
// Description : Simple Translator example from ROSE user manual.
//=========================================================================================

#include "rose.h"
using namespace std;


int main( int argc, char * argv[] )
{
	//Build the AST used by ROSE
	SgProject* sageProject  = frontend (argc , argv) ;
	ROSE_ASSERT (sageProject != NULL);

	// Run internal consistency tests on AST
	//AstTests::runAllTests(sageProject);
	
	// Generate source code from AST and call the vendor's compiler
	return backend(sageProject);
}
