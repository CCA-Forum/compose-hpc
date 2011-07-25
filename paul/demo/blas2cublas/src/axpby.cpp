#include "blas2cublas.h"

using namespace std;

void handleAXPBY(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){

	ostringstream cocciStream;
	string prefix = "";
	string len_X = "";
	string len_Y = "";

	size_t preInd = arrayPrefix.find_first_of(":");
	if(preInd != string::npos) prefix = arrayPrefix.substr(0,preInd);

	size_t lenInd = arrayPrefix.find_last_of(":");
	if(lenInd != string::npos) len_X = arrayPrefix.substr(preInd+1,lenInd-preInd-1);

	len_Y = arrayPrefix.substr(lenInd+1);

	arrayPrefix = prefix;

	string aType = "";
	string blasCall = fname;
	string cublasCall = "";

	SgNode* matrixAptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);
	SgNode* matrixBptr = fArgs->get_traversalSuccessorByIndex(4);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(5);

	string matARef = matrixAptr->unparseToCompleteString();
	string matBRef = matrixBptr->unparseToCompleteString();
	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();

	if(fname.find("saxpby") != string::npos){
		aType = "float";
		cublasCall = "cublasSaxpby";
	}
	else if(fname.find("daxpby") != string::npos){
		aType = "double";
		cublasCall = "cublasDaxpby";
	}
	else if(fname.find("caxpby") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCaxpby";
	}
	else if(fname.find("zaxpby") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZaxpby";
	}

	cocciStream << "@@ \n";
	cocciStream << "expression n, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n, "<<matARef<<","<<vecXRef<<",incx,"<<matBRef<<","<<vecYRef<<",incy); \n";

	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_A;  \n";
	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_B;  \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_A);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_B);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_A,"<<matARef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_B,"<<matBRef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_A, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_B,"<<arrayPrefix<<"_Y,incy);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vector back to host */  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_A); \n";
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_B); \n";
	cocciFptr << cocciStream.str();

}

