#include "blas2cublas.h"

using namespace std;

void handleROT(ofstream &cocciFptr,string fname, string arrayPrefix, SgExprListExp* fArgs){
	
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

	SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
	SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

	SgNode* cPtr = fArgs->get_traversalSuccessorByIndex(5);
	SgNode* sPtr = fArgs->get_traversalSuccessorByIndex(6);

	string vecXRef = vecXptr->unparseToCompleteString();
	string vecYRef = vecYptr->unparseToCompleteString();
	string cRef = cPtr->unparseToCompleteString();
	string sRef = sPtr->unparseToCompleteString();


	if(fname.find("srot") != string::npos){
		aType = "float";
		cublasCall = "cublasSrot";
	}
	else if(fname.find("drot") != string::npos){
		aType = "double";
		cublasCall = "cublasDrot";
	}
	else if(fname.find("crot") != string::npos){
		aType = "cuComplex";
		cublasCall = "cublasCrot";
	}
	else if(fname.find("zrot") != string::npos){
		aType = "cuDoubleComplex";
		cublasCall = "cublasZrot";
	}

	cocciStream << "@@ \n";
	cocciStream << "expression n, incx, incy;  \n";
	cocciStream << "@@ \n";

	cocciStream << "- "<<blasCall<<"(n,"<<vecXRef<<",incx,"<<vecYRef<<",incy,"<<cRef<<","<<sRef<<"); \n";

	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_C;  \n";
	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_S;  \n";
	cocciStream << "+ "<<aType<<" *"<<arrayPrefix<<"_result;  \n";

	DeclareDevicePtrB2(cocciStream,aType,arrayPrefix,false,true,true);

	cocciStream << "+  /* Allocate device memory */  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_X);  \n";
	cocciStream << "+  cublasAlloc(n, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_Y);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_C);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_S);  \n";
	cocciStream << "+  cublasAlloc(1, sizeType_"<<arrayPrefix<<", (void**)&"<<arrayPrefix<<"_result);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy matrix, vectors to device */     \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecXRef<<", incx, "<<arrayPrefix<<"_X, incx);  \n";
	cocciStream << "+  cublasSetVector ( n, sizeType_"<<arrayPrefix<<","<<vecYRef<<", incy, "<<arrayPrefix<<"_Y, incy);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_C,"<<cRef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";
	cocciStream << "+  cudaMemcpy("<<arrayPrefix<<"_S,"<<sRef<<",sizeType_"<<arrayPrefix<<",cudaMemcpyHostToDevice);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* CUBLAS call */  \n";
	cocciStream << "+  "<<cublasCall<<"(n, "<<arrayPrefix<<"_X,incx,"<<arrayPrefix<<"_Y,incy,"<<arrayPrefix<<"_C,"<<arrayPrefix<<"_S);  \n";

	cocciStream << "+  \n";
	cocciStream << "+  /* Copy result vectors back to host */  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_X, incx, "<<vecXRef<<", incx);  \n";
	cocciStream << "+  cublasSetVector (n, sizeType_"<<arrayPrefix<<","<<arrayPrefix<<"_Y, incy, "<<vecYRef<<", incy);  \n";
	FreeDeviceMemoryB2(cocciStream,arrayPrefix,false,true,true);
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_C); \n";
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_S); \n";
	cocciStream << "+  cublasFree("<<arrayPrefix<<"_result); \n";
	cocciFptr << cocciStream.str();

}

