#include "Blas2Cublas.h"

using namespace std;

void handleHESYR2(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
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
    string vecYRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;
    SgNode* vecYptr = NULL;

    if (checkBlasCallType) {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(8);
        vecXptr = fArgs->get_traversalSuccessorByIndex(4);
        vecYptr = fArgs->get_traversalSuccessorByIndex(6);
    }

    else {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        vecXptr = fArgs->get_traversalSuccessorByIndex(3);
        vecYptr = fArgs->get_traversalSuccessorByIndex(5);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();
    vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("cher2") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCher2";
    } else if (fname.find("zher2") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZher2";
    } else if (fname.find("ssyr2") != string::npos) {
        aType = "float";
        cublasCall = "cublasSsyr2";
    } else if (fname.find("dsyr2") != string::npos) {
        aType = "double";
        cublasCall = "cublasDsyr2";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo;  \n";
    cocciStream << "expression n, alpha, lda, incx, incy;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo, n, alpha," << vecXRef
                << ",incx," << vecYRef << ",incy," << matARef << ",lda); \n";
    else
        cocciStream << "- " << blasCall << "(uplo, n, alpha," << vecXRef
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
        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_A, n*n * sizeType_" << uPrefix
                << ");  \n";
        arrName = uPrefix+"_A";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_X, n * sizeType_" << uPrefix << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, n * sizeType_" << uPrefix << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", n, (void *) " << uPrefix
                << "_A, n);  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecYRef << ", incy, " << uPrefix << "_Y, incy);  \n\n";
        arrName = uPrefix+"_Y";
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
                << uPrefix << "_X,incx," << uPrefix << "_Y,incy," << uPrefix
                << "_A,lda);  \n\n";
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
                << ");  \n";
        arrName = uPrefix+"_X";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+ " << cudaStat << " = cudaMalloc((void**)&" << uPrefix << "_Y, *(n) * sizeType_" << uPrefix
                << ");  \n\n";
        arrName = uPrefix+"_Y";
        memAllocCheck(cocciStream, arrName);

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+ " << stat << " = cublasSetMatrix ( *(n),*(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(n), (void *) " << uPrefix
                << "_A, *(n));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n";
        arrName = uPrefix+"_X";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ " << stat << " = cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecYRef << ", *(incy), " << uPrefix << "_Y, *(incy));  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);


        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(uplo),*(n), *(alpha),"
                << uPrefix << "_X,*(incx)," << uPrefix << "_Y,*(incy),"
                << uPrefix << "_A, *(lda));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetMatrix ( *(n),*(n), sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, *(n), (void *) " << matARef
                << ", *(n));  \n";
        arrName = uPrefix+"_A";
        memCpyCheck(cocciStream, arrName);
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

