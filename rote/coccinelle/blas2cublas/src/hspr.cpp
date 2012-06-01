#include "Blas2Cublas.h"

using namespace std;

void handleHSPR(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs, int *firstBlas) {

    ostringstream cocciStream;

    string matARef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string cbTrans = "";
    string cblasUplo = "";
    string uplo = "";
    string vecXRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;

    if (checkBlasCallType) {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
        vecXptr = fArgs->get_traversalSuccessorByIndex(4);
    }

    else {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
        vecXptr = fArgs->get_traversalSuccessorByIndex(3);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();

    if (fname.find("chpr") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasChpr";
    } else if (fname.find("zhpr") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZhpr";
    } else if (fname.find("sspr") != string::npos) {
        aType = "float";
        cublasCall = "cublasSspr";
    } else if (fname.find("dspr") != string::npos) {
        aType = "double";
        cublasCall = "cublasDspr";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo;  \n";
    cocciStream << "expression n, alpha, incx;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo, n, alpha," << vecXRef
                << ",incx," << matARef << "); \n";
    else
        cocciStream << "- " << blasCall << "(uplo, n, alpha," << vecXRef
                << ",incx," << matARef << "); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, false);
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
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, n*n * sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", n, (void *) " << uPrefix
                << "_A, n);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);


        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);

        if (cblasUplo == "CblasUpper")
            uplo = "CUBLAS_FILL_MODE_UPPER";
        else if (cblasUplo == "CblasLower")
            uplo = "CUBLAS_FILL_MODE_LOWER";
        else {
            uplo = uPrefix + "_uplo";
            cocciStream << "+ char " << uplo << "; \n";
            cocciStream << "+ if(" << cblasUplo << " == CblasUpper) " << uplo
                    << " = CUBLAS_FILL_MODE_UPPER; \n";
            cocciStream << "+ else " << uplo << " = CUBLAS_FILL_MODE_LOWER; \n";

        }

        cocciStream << "+  " << stat << " = " << cublasCall << "(" << uplo << ",n, &"<<alpha<<","
                << uPrefix << "_X,incx," << uPrefix << "_A);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, n, (void *) " << matARef
                << ", n);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);
    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, *(n) * *(n) * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);


        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(n),*(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(n), (void *) " << uPrefix
                << "_A, *(n));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n\n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);


        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(uplo), *(n), *(alpha),"
                << uPrefix << "_X,*(incx)," << uPrefix << "_A);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix ( *(n), *(n), sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, *(n), (void *) " << matARef
                << ", *(n));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, false);
    cocciFptr << cocciStream.str();

}

