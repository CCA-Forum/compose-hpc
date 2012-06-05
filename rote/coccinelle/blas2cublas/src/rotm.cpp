#include "Blas2Cublas.h"

using namespace std;

void handleROTM(ofstream &cocciFptr, bool checkBlasCallType, string fname,
        string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;

    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
    SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

    SgNode* paramPtr = fArgs->get_traversalSuccessorByIndex(5);

    string vecXRef = vecXptr->unparseToCompleteString();
    string vecYRef = vecYptr->unparseToCompleteString();
    string paramRef = paramPtr->unparseToCompleteString();

    if (fname.find("srotm") != string::npos) {
        aType = "float";
        cublasCall = "cublasSrotm";
    } else if (fname.find("drotm") != string::npos) {
        aType = "double";
        cublasCall = "cublasDrotm";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression n, incx, incy;  \n";
    cocciStream << "@@ \n";

    cocciStream << "- " << blasCall << "(n," << vecXRef << ",incx," << vecYRef
            << ",incy," << paramRef << "); \n";

    cocciStream << "+ " << aType << " *" << uPrefix << "_param;  \n";
    cocciStream << "+ " << aType << " *" << uPrefix << "_result;  \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, false, true, true);

    if (checkBlasCallType) {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_Y);  \n";
        cocciStream << "+  cublasAlloc(5, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_param);  \n";
        cocciStream << "+  cublasAlloc(1, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_result);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n";
        cocciStream << "+  cudaMemcpy(" << uPrefix << "_param," << paramRef
                << ",5*sizeof(" << aType << "),cudaMemcpyHostToDevice);  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(n, " << uPrefix << "_X,incx,"
                << uPrefix << "_Y,incy," << uPrefix << "_param);  \n\n";

        cocciStream << "+  /* Copy result vectors back to host */  \n";
        cocciStream << "+  cublasSetVector (n, sizeType_" << uPrefix << ","
                << uPrefix << "_X, incx, " << vecXRef << ", incx);  \n";
        cocciStream << "+  cublasSetVector (n, sizeType_" << uPrefix << ","
                << uPrefix << "_Y, incy, " << vecYRef << ", incy);  \n";
    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_Y);  \n";
        cocciStream << "+  cublasAlloc(5, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_param);  \n";
        cocciStream << "+  cublasAlloc(1, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_result);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n";
        cocciStream << "+  cudaMemcpy(" << uPrefix << "_param," << paramRef
                << ",5*sizeof(" << aType << "),cudaMemcpyHostToDevice);  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(*(n), " << uPrefix
                << "_X,*(incx)," << uPrefix << "_Y,*(incy)," << uPrefix
                << "_param);  \n\n";

        cocciStream << "+  /* Copy result vectors back to host */  \n";
        cocciStream << "+  cublasSetVector (*(n), sizeType_" << uPrefix << ","
                << uPrefix << "_X, *(incx), " << vecXRef << ", *(incx));  \n";
        cocciStream << "+  cublasSetVector (*(n), sizeType_" << uPrefix << ","
                << uPrefix << "_Y, *(incy), " << vecYRef << ", *(incy));  \n";
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, false, true, true);
    cocciStream << "+ cublasFree(" << uPrefix << "_param);\n";
    cocciStream << "+ cublasFree(" << uPrefix << "_result); \n";
    cocciFptr << cocciStream.str();

}

