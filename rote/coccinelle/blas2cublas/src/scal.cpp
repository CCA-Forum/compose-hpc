#include "Blas2Cublas.h"

using namespace std;

void handleSCAL(ofstream &cocciFptr, bool checkBlasCallType, string fname,
        string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;

    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(2);
    string vecXRef = vecXptr->unparseToCompleteString();

    if (fname.find("sscal") != string::npos) {
        aType = "float";
        cublasCall = "cublasSscal";
    } else if (fname.find("dscal") != string::npos) {
        aType = "double";
        cublasCall = "cublasDscal";
    } else if (fname.find("cscal") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCscal";
    } else if (fname.find("zscal") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZscal";
    } else if (fname.find("csscal") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCsscal";
    } else if (fname.find("zdscal") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZdscal";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression n, a, incx;  \n";
    cocciStream << "@@ \n";

    cocciStream << "- " << blasCall << "(n, a," << vecXRef << ",incx); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, false, true, false);
    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string stat = "CudaStatReturn";

    if (*firstBlas == 1) {

        cocciStream << "+ cublasHandle_t " << handle << "; \n";
        cocciStream << "+ cublasStatus_t " << stat << " = cublasCreate(&"
                << handle << "); \n";
        cocciStream << "+ cudaError_t " << cudaStat << "; \n";
        cocciStream << "+  \n";
        cocciStream << "+ if( " << stat << " != CUBLAS_STATUS_SUCCESS ) { \n";
        cocciStream
                << "+        printf ( \"CUBLAS initialization failed \\n\" ); \n";
        cocciStream << "+        return EXIT_FAILURE; \n";
        cocciStream << "+  } \n\n";
        cocciStream << "+  \n";
        cocciStream << "+ // Move and uncomment the following handle destroy call to the end of your cuda code. \n";
        cocciStream << "+ // cublasDestroy(&" << handle << "); \n";
        cocciStream << "+  \n";
    }

    string arrName="";

    if (checkBlasCallType) {
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&"<< uPrefix << "_X, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(n, a, " << uPrefix
                << "_X,incx);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector (n, sizeType_" << uPrefix << ","
                << uPrefix << "_X, incx, " << vecXRef << ", incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);
    }

    else {
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);


        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(n), *(a), " << uPrefix
                << "_X,*(incx));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector (*(n), sizeType_" << uPrefix << ","
                << uPrefix << "_X, *(incx), " << vecXRef << ", *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, false, true, false);
    cocciFptr << cocciStream.str();
}

