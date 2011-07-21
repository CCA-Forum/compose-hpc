#include "Transforms.h"

#include "blas2cublas/src/blas2cublas.h"

using namespace SageInterface;

void handleBlasCalls(ofstream&, string&, SgExprListExp *, string);
void cublasHeaderInsert(ofstream&);

bool fileExists(const std::string& filename)
{
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}

Transform::Transform(SgLocatedNode *theroot) {
	root = theroot;
}

Transform *Transform::get_transform(SgLocatedNode *theroot,Annotation *ann) {
	if(ann->get_id() == "ABSORB_STRUCT_ARRAY") {
		return new AbsorbStructTransform(ann,theroot);
	}
	else if(ann->get_id() == "BLAS_TO_CUBLAS") {
		return new BlasToCublasTransform(ann,theroot);
	}
	else {
		cerr << "Unknown annotation: " << ann->get_id()  << endl;
		exit(1);
	}
}

//To handle BLAS to CUBLAS transformations

BlasToCublasTransform::BlasToCublasTransform(Annotation *a,SgLocatedNode *p)
: Transform(p) {
	arrayPrefix = a->get_attrib("prefix")->string_value();
	if(arrayPrefix.length() == 0) {
		cerr << "BLAS To CUBLAS transformation prefix not specified. " << endl;
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

void AbsorbStructTransform::generate(string inpFile, int *fileCount) {
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




void BlasToCublasTransform::generate(string inpFile, int *fileCount){

	SgExprStatement *es = isSgExprStatement(root);
	SgExpression *fCall = es->get_expression();
	SgFunctionCallExp *funCall = isSgFunctionCallExp(fCall);
	SgFunctionSymbol *funcSym = funCall->getAssociatedFunctionSymbol();
	SgName funcName = funcSym->get_name();
	string &fname = (&funcName)->getString();

	SgExprListExp *fArgs = funCall->get_args();
	size_t nArgs = fArgs->get_numberOfTraversalSuccessors();

	string cocciFile = inpFile + "_blasCalls.cocci";

	ofstream cocciFptr;

	if(!fileExists(cocciFile) || *fileCount == 0)
	{
		//So create the file
		cocciFptr.open(cocciFile.c_str());
		cublasHeaderInsert(cocciFptr);
		*fileCount = -1;
	}

	else{
		cocciFptr.open(cocciFile.c_str(), ios::app);
		cocciFptr << "\n\n\n";
	}

	handleBlasCalls(cocciFptr,fname,fArgs,arrayPrefix);

	if(cocciFptr.is_open()) cocciFptr.close();
}




void handleBlasCalls(ofstream &cocciFptr,string &fname,SgExprListExp *fArgs, string arrayPrefix){

	size_t npos = string::npos;

	bool warnRowMajor = true;

	bool checkBlasCallType = (fname.find("cblas") != npos);

	if(checkBlasCallType){
		string cblasOrder  = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		if(cblasOrder == "CblasColMajor") warnRowMajor = false;
	}
	else warnRowMajor = false;

	/* --------------- BLAS 3 CALLS -----------------*/

	if(fname.find("gemm") != npos) handleGEMM(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("symm") != npos || fname.find("hemm") != npos)
		handleSYHEMM(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);


	else if(fname.find("herk") != npos || fname.find("syrk") != npos)
		handleSYHERK(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("her2k") != npos || fname.find("syr2k") != npos)

		handleSYHER2K(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("trsm") != npos || fname.find("trmm") != npos)

		handleTRSMM(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	/* --------------- BLAS 2 CALLS -----------------*/

	else if(fname.find("gbmv") != npos) 	handleGBMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("gemv") != npos) handleGEMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("ger") != npos) handleGER(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("hbmv") != npos || fname.find("sbmv") != npos)
		handleHSBMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("hemv") != npos || fname.find("symv") != npos)
		handleHSEYMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("her2") != npos || fname.find("syr2") != npos)
		handleHESYR2(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("her") != npos || fname.find("syr") != npos)
		handleHESYR(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("hpmv") != npos || fname.find("spmv") != npos)
		handleHSPMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("hpr2") != npos || fname.find("spr2") != npos)
		handleHSPR2(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("hpr") != npos || fname.find("spr") != npos)
		handleHSPR(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("tbmv") != npos || fname.find("tbsv") != npos)
		handleTBSMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("tpmv") != npos || fname.find("tpsv") != npos)
		handleTPSMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);

	else if(fname.find("trmv") != npos || fname.find("trsv") != npos)
		handleTRSMV(cocciFptr,checkBlasCallType,warnRowMajor,fname,arrayPrefix,fArgs);


	/* --------------- BLAS 1 CALLS -----------------*/

	else if(fname.find("asum") != npos || fname.find("nrm2") != npos ||
		fname.find("amin") != npos || fname.find("amax") != npos)

		; //handleSumNrm2Aminmax(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("axpy") != npos) handleAXPY(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("axpby") != npos) handleAXPBY(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("copy") != npos) handleCOPY(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("dot") != npos)  ; //handleDOT(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("scal") != npos) handleSCAL(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("swap") != npos) handleSWAP(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("rotg") != npos || fname.find("rotmg") != npos) ; //do nothing

	else if(fname.find("rotm") != npos) ;//handleROTM(cocciFptr,fname,arrayPrefix,fArgs);

	else if(fname.find("rot") != npos) ; //handleROT(cocciFptr,fname,arrayPrefix,fArgs);

	else cerr << "Unknown BLAS call: " <<fname<< " ...\nNOTE: PAUL does not handle Sparse BLAS calls." << endl;

}



void cublasHeaderInsert(ofstream &cocciFptr){
	cocciFptr << "//Begin Header insertion patch.\n";
	cocciFptr << "//This patch is taken from coccinelle.\n";
	cocciFptr << "//demos file first.cocci\n\n";

	cocciFptr << "@initialize:python@ \n\n";

	cocciFptr << "first = 0 \n";
	cocciFptr << "second = 0 \n\n";

	cocciFptr << "@first_hdr@ \n";
	cocciFptr << "position p; \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "#include <...>@p \n\n";

	cocciFptr << "@script:python@ \n";
	cocciFptr << "p << first_hdr.p; \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "if first == 0: \n";
	cocciFptr << "   print \"keeping first hdr %s\" % (p[0].line) \n";
	cocciFptr << "   first = int(p[0].line) \n";
	cocciFptr << "else: \n";
	cocciFptr << "  print \"dropping first hdr\" \n";
	cocciFptr << "  cocci.include_match(False) \n";

	cocciFptr << "@second_hdr@ \n";
	cocciFptr << "position p; \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "#include \"...\"@p \n\n";

	cocciFptr << "@script:python@ \n";
	cocciFptr << "p << second_hdr.p; \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "if int(p[0].line) > first and first != 0: \n";
	cocciFptr << "   print \"dropping second hdr\" \n";
	cocciFptr << "   cocci.include_match(False) \n";
	cocciFptr << "else: \n";
	cocciFptr << "  if second == 0: \n";
	cocciFptr << "     print \"keeping second hdr %s because of %d\" % (p[0].line,first) \n";
	cocciFptr << "     second = int(p[0].line) \n";
	cocciFptr << "  else: \n";
	cocciFptr << "     print \"dropping second hdr\" \n";
	cocciFptr << "     cocci.include_match(False) \n\n";

	cocciFptr << "@done@ \n";
	cocciFptr << "position second_hdr.p; \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "+#include \"cublas.h\" \n";
	cocciFptr << "#include \"...\"@p \n\n";

	cocciFptr << "@depends on never done@ \n";
	cocciFptr << "@@ \n\n";

	cocciFptr << "+#include \"cublas.h\" \n";
	cocciFptr << "#include <...> \n\n";

	cocciFptr << "//End Header insertion patch.\n\n";
}


