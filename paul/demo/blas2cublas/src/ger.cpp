#include "blas2cublas.h"

using namespace std;

void handleGER(ofstream &cocciFptr,bool checkBlasCallType, bool warnRowMajor, string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	string prefix = "";
	string len_X = "m";
	string len_Y = "n";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	len_Y = arrayPrefix.substr(lenInd+1);

	arrayPrefix = prefix;

	string matARef = "";
	string aType = "";
	string blasCall = fname;
	string cublasCall = "";
	string cbTrans="";
	string cblasTrans = "";

	string vecXRef="";
	string vecYRef="";

	SgNode* matrixAptr = NULL;
	SgNode* vecXptr = NULL;
	SgNode* vecYptr = NULL;

	if(checkBlasCallType){
		cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
		vecXptr = fArgs->get_traversalSuccessorByIndex(4);
		vecYptr = fArgs->get_traversalSuccessorByIndex(6);
	}

	else {
		cblasTrans = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
		matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
		vecXptr = fArgs->get_traversalSuccessorByIndex(3);
		vecYptr = fArgs->get_traversalSuccessorByIndex(5);

	}

	if(    cblasTrans  == "CblasTrans")     cbTrans = "\'T\'";
	else if(cblasTrans == "CblasNoTrans")   cbTrans = "\'N\'";
	else if(cblasTrans == "CblasConjTrans") cbTrans = "\'C\'";

	matARef = matrixAptr->unparseToCompleteString();
	vecXRef = vecXptr->unparseToCompleteString();
	vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("cgerc") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgerc";
	}
	else if(fname.find("zgerc") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgerc";
	}
	else if(fname.find("cgeru") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCgeru";
	}
	else if(fname.find("zgeru") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZgeru";
	}
	else if(fname.find("sger") != string::npos){
		aType = "float";
		cublasCall = "cublasSger";
	}
	else if(fname.find("dger") != string::npos){
		aType = "double";
		cublasCall = "cublasDger";
	}

	cocciFptr << "@@ \n";
	cocciFptr << "identifier order;  \n";
	cocciFptr << "expression m, n, alpha, incx, incy, lda;  \n";
	cocciFptr << "@@ \n";

	if(checkBlasCallType)
		cocciFptr << "- "<<blasCall<<"(order,m, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<< ",lda); \n";
	else cocciFptr << "- "<<blasCall<<"(m, n, alpha,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<matARef<< ",lda); \n";

	DeclareDevicePtrB2(cocciFptr,aType,arrayPrefix,true,true,true);

	cocciFptr << "+  /* Allocate device memory */  \n";
	cocciFptr << "+  cublasAlloc(m*n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciFptr << "+  cublasAlloc("<<len_X<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciFptr << "+  cublasAlloc("<<len_Y<<", sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy matrix, vectors to device */     \n";
	cocciFptr << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<matARef<<", m, (void *) "<<arrayPrefix<<"_A, m);  \n";
	cocciFptr << "+  cublasSetVector ( len_X, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciFptr << "+  cublasSetVector ( len_Y, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* CUBLAS call */  \n";
	RowMajorWarning(cocciFptr,warnRowMajor);


	cocciFptr << "+  "<<cublasCall<<"(m, n, alpha,"<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy,"<<arrayPrefix<<"_A,lda);  \n";

	cocciFptr << "+  \n";
	cocciFptr << "+  /* Copy result matrix back to host */  \n";
	cocciFptr << "+  cublasSetMatrix ( m, n, sizeType_"<<arrayPrefix<<", (void *)"<<arrayPrefix<<"_A, m, (void *) "<<matARef<<", m);  \n";
	FreeDeviceMemoryB2(cocciFptr,arrayPrefix,true,true,true);

}

