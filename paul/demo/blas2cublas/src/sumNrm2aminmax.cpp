#include "blas2cublas.h"

using namespace std;

void handleSumNrm2Aminmax(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	string prefix = "";
	string len_X = "";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	arrayPrefix = prefix;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";
	string cbTrans="";

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
	string vecXRef = vecXptr->unparseToCompleteString();

	if(fname.find("sasum") != string::npos){
		aType = "float";
		cublasCall = "cublasSasum";
	}
	else if(fname.find("dasum") != string::npos){
		aType = "double";
		cublasCall = "cublasDasum";
	}
	else if(fname.find("scasum") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasScasum";
	}
	else if(fname.find("dzasum") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasDzasum";
	}
	else if(fname.find("snrm2") != string::npos){
		aType = "float";
		cublasCall = "cublasSnrm2";
	}
	else if(fname.find("dnrm2") != string::npos){
		aType = "double";
		cublasCall = "cublasDnrm2";
	}
	else if(fname.find("scnrm2") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasScnrm2";
	}
	else if(fname.find("dznrm2") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasDznrm2";
	}
	else if(fname.find("isamin") != string::npos){
		aType = "float";
		cublasCall = "cublasIsamin";
	}
	else if(fname.find("idamin") != string::npos){
		aType = "double";
		cublasCall = "cublasIdamin";
	}
	else if(fname.find("icamin") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasIcamin";
	}
	else if(fname.find("izamin") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasIzamin";
	}
	else if(fname.find("isamax") != string::npos){
		aType = "float";
		cublasCall = "cublasIsamax";
	}
	else if(fname.find("idamax") != string::npos){
		aType = "double";
		cublasCall = "cublasIdamax";
	}
	else if(fname.find("icamax") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasIcamax";
	}
	else if(fname.find("izamax") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasIzamax";
	}


	cocciFptr << "@@ \n";
	cocciFptr << "expression n, incx;  \n";
	cocciFptr << "@@ \n";

	cocciFptr << "<...\n- "<<blasCall<<"(n, "<<vecXRef<<",incx); \n";
	cocciFptr << "+ "<<aType<<" *"<<arrayPrefix<<"_result;  \n";
	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,false,true,false);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_result);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy vector to device */     \n";
	cocciFptr << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	cocciFptr << "+  "<<cublasCall<<"(n,"<<arrayPrefix<<"_X,incx);  \n...>\n";
	cocciFptr << "+  \n";

	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,false,true,false);
	cocciFptr << "+  cublasFree("<<arrayPrefix<<"_result); \n";

}

