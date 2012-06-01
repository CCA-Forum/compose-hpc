#include "Blas2Cublas.h"

using namespace std;

void handleHSEYMV(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
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
        matrixAptr = fArgs->get_traversalSuccessorByIndex(4);
        vecXptr = fArgs->get_traversalSuccessorByIndex(6);
        vecYptr = fArgs->get_traversalSuccessorByIndex(9);
    }

    else {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(3);
        vecXptr = fArgs->get_traversalSuccessorByIndex(5);
        vecYptr = fArgs->get_traversalSuccessorByIndex(8);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();
    vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("chemv") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasChemv";
    } else if (fname.find("zhemv") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZhemv";
    } else if (fname.find("ssymv") != string::npos) {
        aType = "float";
        cublasCall = "cublasSsymv";
    } else if (fname.find("dsymv") != string::npos) {
        aType = "double";
        cublasCall = "cublasDsymv";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo;  \n";
    cocciStream << "expression n, alpha, lda, incx, beta, incy;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo, n, alpha," << matARef
                << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";
    else
        cocciStream << "- " << blasCall << "(uplo, n, alpha," << matARef
                << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, true);
    string handle = "CublasHandle";
    string cudaStat = "CudaStat";
    string alpha = "alpha_" + uPrefix;
    string beta = "beta_" + uPrefix;
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
    cocciStream << "+  "<<aType<<" " << beta << " = beta; \n";
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

        cocciStream << "+  if(beta != 0) {\n" << stat << " = cublasSetVector ( n, sizeType_"
                << uPrefix << "," << vecYRef << ", incy, " << uPrefix
                << "_Y, incy);  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+ }\n";

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
                << uPrefix << "_A,lda," << uPrefix << "_X,incx,&"<<beta<<"," << uPrefix
                << "_Y,incy);  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( n, sizeType_" << uPrefix << ","
                << uPrefix << "_Y, incy, " << vecYRef << ", incy);  \n";
        arrName = uPrefix+"_Y";
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

        cocciStream << "+  if(*(beta) != 0){\n " << stat << " = cublasSetVector ( *(n), sizeType_"
                << uPrefix << "," << vecYRef << ", *(incy), " << uPrefix
                << "_Y, *(incy));  \n\n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

        cocciStream << "+  }  \n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << stat << " = " << cublasCall << "(*(uplo),*(n), *(alpha),"
                << uPrefix << "_A, *(lda)," << uPrefix << "_X,*(incx),*(beta),"
                << uPrefix << "_Y,*(incy));  \n\n";
        blasSuccessCheck(cocciStream,cublasCall);

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+ " << stat << " = cublasGetVector ( *(n), sizeType_" << uPrefix << ","
                << uPrefix << "_Y, *(incy), " << vecYRef << ", *(incy));  \n";
        arrName = uPrefix+"_Y";
        memCpyCheck(cocciStream, arrName);

    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

