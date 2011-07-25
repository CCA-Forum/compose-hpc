#include "blas2cublas.h"

using namespace std;

void handleSCAL(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
	ostringstream cocciStream;
	string prefix = "";
	string len_X = "";
	string len_Y = "";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	arrayPrefix = prefix;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* matrixAptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);

	string matARef = matrixAptr->unparseToCompleteString();
	string vecXRef = vecXptr->unparseToCompleteString();

	if(fname.find("sscal") != string::npos){
		aType = "float";
		cublasCall = "cublasSscal";
	}
	else if(fname.find("dscal") != string::npos){
		aType = "double";
		cublasCall = "cublasDscal";
	}
	else if(fname.find("cscal") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCscal";
	}
	else if(fname.find("zscal") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZscal";
	}
	else if(fname.find("csscal") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCsscal";
	}
	else if(fname.find("zdscal") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZdscal";
	}

	cocciStream << "@@ \n";
	cocciStream << "expression n, incx;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, "<<matARef<<","<<vecXRef<<",incx); \n";

	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,false);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_A,"<<matARef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";


	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_A, "<<arrayPrefix<<"_X,incx);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,false);
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_A); \n";
	cocciFptr << cocciStream.str();
}

