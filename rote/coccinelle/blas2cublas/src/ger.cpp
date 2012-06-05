#include "Blas2Cublas.h"

using namespace std;

void handleGER(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;

    string matARef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string vecXRef = "";
    string vecYRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;
    SgNode* vecYptr = NULL;

    if (checkBlasCallType) {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
        vecXptr = fArgs->get_traversalSuccessorByIndex(4);
        vecYptr = fArgs->get_traversalSuccessorByIndex(6);
    }

    else {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        vecXptr = fArgs->get_traversalSuccessorByIndex(3);
        vecYptr = fArgs->get_traversalSuccessorByIndex(5);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();
    vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("cgerc") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgerc";
    } else if (fname.find("zgerc") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgerc";
    } else if (fname.find("cgeru") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgeru";
    } else if (fname.find("zgeru") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgeru";
    } else if (fname.find("sger") != string::npos) {
        aType = "float";
        cublasCall = "cublasSger";
    } else if (fname.find("dger") != string::npos) {
        aType = "double";
        cublasCall = "cublasDger";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order;  \n";
    cocciStream << "expression m, n, alpha, incx, incy, lda;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,m, n, alpha," << vecXRef
                << ",incx," << vecYRef << ",incy," << matARef << ",lda); \n";
    else
        cocciStream << "- " << blasCall << "(m, n, alpha," << vecXRef
                << ",incx," << vecYRef << ",incy," << matARef << ",lda); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, true);
    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string alpha = "alpha_" + uPrefix;
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

    cocciStream << "+  "<<aType<<" "<< alpha << " = alpha; \n";
    cocciStream << "+  \n";
    string arrName = "";

    if (checkBlasCallType) {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, m*n * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&"
                << uPrefix << "_X, m * sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&"
                << uPrefix << "_Y, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", m, (void *) " << uPrefix
                << "_A, m);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( m, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);

        cocciStream << "+  " << stat << " = " << cublasCall << "(m, n, &"<<alpha<<"," << uPrefix
                << "_X,incx," << uPrefix << "_Y,incy," << uPrefix
                << "_A,lda);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, m, (void *) " << matARef
                << ", m);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);
    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, *(m) * *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, *(m) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(m), (void *) " << uPrefix
                << "_A, *(m));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(m), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);


        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(m), *(n), *(alpha),"
                << uPrefix << "_X,*(incx)," << uPrefix << "_Y,*(incy),"
                << uPrefix << "_A,*(lda));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, *(m), (void *) " << matARef
                << ", *(m));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

