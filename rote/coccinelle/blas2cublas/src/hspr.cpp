#include "Blas2Cublas.h"

using namespace std;

void handleHSPR(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs) {

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

    if (checkBlasCallType) {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(n*n, sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(n, sizeType_" << uPrefix << ", (void**)&"
                << uPrefix << "_X);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", n, (void *) " << uPrefix
                << "_A, n);  \n";
        cocciStream << "+  cublasSetVector ( n, sizeType_" << uPrefix << ","
                << vecXRef << ", incx, " << uPrefix << "_X, incx);  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);

        if (cblasUplo == "CblasUpper")
            uplo = "\'U\'";
        else if (cblasUplo == "CblasLower")
            uplo = "\'L\'";
        else {
            uplo = uPrefix + "_uplo";
            cocciStream << "+ char " << uplo << "; \n";
            cocciStream << "+ if(" << cblasUplo << " == CblasUpper) " << uplo
                    << " = \'U\'; \n";
            cocciStream << "+ else " << uplo << " = \'L\'; \n";

        }

        cocciStream << "+  " << cublasCall << "(" << uplo << ",n, alpha,"
                << uPrefix << "_X,incx," << uPrefix << "_A);  \n\n";
        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+  cublasSetMatrix ( n,n, sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, n, (void *) " << matARef
                << ", n);  \n";
    }

    else {

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(*(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n\n";

        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( *(n),*(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(n), (void *) " << uPrefix
                << "_A, *(n));  \n";
        cocciStream << "+  cublasSetVector ( *(n), sizeType_" << uPrefix << ","
                << vecXRef << ", *(incx), " << uPrefix << "_X, *(incx));  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(*(uplo), *(n), *(alpha),"
                << uPrefix << "_X,*(incx)," << uPrefix << "_A);  \n\n";
        cocciStream << "+  /* Copy result matrix back to host */  \n";
        cocciStream << "+  cublasSetMatrix ( *(n), *(n), sizeType_" << uPrefix
                << ", (void *)" << uPrefix << "_A, *(n), (void *) " << matARef
                << ", *(n));  \n";
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, false);
    cocciFptr << cocciStream.str();

}

