#include "Blas2Cublas.h"

using namespace std;

void handleSWAP(ofstream &cocciFptr, bool checkBlasCallType, string fname,
        string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;

    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    SgNode* vecXptr = fArgs->get_traversalSuccessorByIndex(1);
    SgNode* vecYptr = fArgs->get_traversalSuccessorByIndex(3);

    string vecXRef = vecXptr->unparseToCompleteString();
    string vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("sswap") != string::npos) {
        aType = "float";
        cublasCall = "cublasSswap";
    } else if (fname.find("dswap") != string::npos) {
        aType = "double";
        cublasCall = "cublasDswap";
    } else if (fname.find("cswap") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCswap";
    } else if (fname.find("zswap") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZswap";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression n, incx, incy;  \n";
    cocciStream << "@@ \n";

    cocciStream << "- " << blasCall << "(n, " << vecXRef << ",incx," << vecYRef
            << ",incy); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, false, true, true);
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
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, n * sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(n, " << uPrefix << "_X,incx,"
                << uPrefix << "_Y,incy);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy swapped vectors back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector (n, sizeType_" << uPrefix << ","
                << uPrefix << "_X, incx, " << vecXRef << ", incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasGetVector (n, sizeType_" << uPrefix << ","
                << uPrefix << "_Y, incy, " << vecYRef << ", incy);  \n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);
    }

    else {
        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(n), " << uPrefix
                << "_X,*(incx)," << uPrefix << "_Y,*(incy));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy swapped vectors back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector (*(n), sizeType_" << uPrefix << ","
                << uPrefix << "_X, *(incx), " << vecXRef << ", *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasGetVector (*(n), sizeType_" << uPrefix << ","
                << uPrefix << "_Y, *(incy), " << vecYRef << ", *(incy));  \n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, false, true, true);
    cocciFptr << cocciStream.str();

}

