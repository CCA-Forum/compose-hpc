#include "Blas2Cublas.h"

using namespace std;

void handleGBMV(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs) {

    string lenXY = "";

    ostringstream cocciStream;

    string matARef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";

    string cbTrans = "";
    string cblasTrans = "";
    string vecXRef = "";
    string vecYRef = "";

    SgNode* matrixAptr = NULL;
    SgNode* vecXptr = NULL;
    SgNode* vecYptr = NULL;

    if (checkBlasCallType) {
        cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(7);
        vecXptr = fArgs->get_traversalSuccessorByIndex(9);
        vecYptr = fArgs->get_traversalSuccessorByIndex(12);
    }

    else {
        cblasTrans = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
        vecXptr = fArgs->get_traversalSuccessorByIndex(8);
        vecYptr = fArgs->get_traversalSuccessorByIndex(11);

    }

    matARef = matrixAptr->unparseToCompleteString();
    vecXRef = vecXptr->unparseToCompleteString();
    vecYRef = vecYptr->unparseToCompleteString();

    if (fname.find("sgbmv") != string::npos) {
        aType = "float";
        cublasCall = "cublasSgbmv";
    } else if (fname.find("dgbmv") != string::npos) {
        aType = "double";
        cublasCall = "cublasDgbmv";
    } else if (fname.find("cgbmv") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCgbmv";
    } else if (fname.find("zgbmv") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZgbmv";
    }

    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,trans;  \n";
    cocciStream
            << "expression m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy;  \n";
    cocciStream << "@@ \n";

    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,trans,m, n, kl, ku, alpha,"
                << matARef << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";
    else
        cocciStream << "- " << blasCall << "(trans,m, n, kl, ku, alpha,"
                << matARef << ",lda," << vecXRef << ",incx,beta," << vecYRef
                << ",incy); \n";

    DeclareDevicePtrB2(cocciStream, aType, uPrefix, true, true, true);

    if (checkBlasCallType) {

        if (isRowMajor) {
            if (cblasTrans == "CblasTrans") {
                cbTrans = "\'N\'";
                lenXY = "n";
            } else if (cblasTrans == "CblasNoTrans") {
                cbTrans = "\'T\'";
                lenXY = "m";
            } else if (cblasTrans == "CblasConjTrans") {
                cbTrans = "\'C\'";
                lenXY = "m";
            } else {
                cbTrans = uPrefix + "_trans";
                lenXY = uPrefix + "_lenXY";
                cocciStream << "+ int " << lenXY << "; \n";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = \'N\'; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = \'T\'; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = \'C\'; \n\n";
                cocciStream << "+ if(" << cbTrans << " == \'N\') " << lenXY
                        << " = n; \n";
                cocciStream << "+ else " << lenXY << " = m; \n\n";

            }
        } else {
            if (cblasTrans == "CblasTrans") {
                cbTrans = "\'T\'";
                lenXY = "m";
            } else if (cblasTrans == "CblasNoTrans") {
                cbTrans = "\'N\'";
                lenXY = "n";
            } else if (cblasTrans == "CblasConjTrans") {
                cbTrans = "\'C\'";
                lenXY = "m";
            } else {
                cbTrans = uPrefix + "_trans";
                lenXY = uPrefix + "_lenXY";
                cocciStream << "+ int " << lenXY << "; \n";
                cocciStream << "+ char " << cbTrans << "; \n";
                cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                        << cbTrans << " = \'T\'; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasNoTrans) " << cbTrans << " = \'N\'; \n";
                cocciStream << "+ else if(" << cblasTrans
                        << " == CblasConjTrans) " << cbTrans
                        << " = \'C\'; \n\n";

                cocciStream << "+ if(" << cbTrans << " == \'N\') " << lenXY
                        << " = n; \n";
                cocciStream << "+ else " << lenXY << " = m; \n\n";

            }
        }

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(m*n, sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(" << lenXY << ", sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(" << lenXY << ", sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_Y);  \n\n";
        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( m, n, sizeType_" << uPrefix
                << ", (void *)" << matARef << ", m, (void *) " << uPrefix
                << "_A, m);  \n";
        cocciStream << "+  cublasSetVector ( " << lenXY << ",, sizeType_"
                << uPrefix << "," << vecXRef << ", incx, " << uPrefix
                << "_X, incx);  \n";
        cocciStream << "+  if(beta != 0) cublasSetVector (" << lenXY
                << ", sizeType_" << uPrefix << "," << vecYRef << ", incy, "
                << uPrefix << "_Y, incy);  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall << "(" << cbTrans
                << ",m, n, kl, ku, alpha," << uPrefix << "_A,lda," << uPrefix
                << "_X,incx,beta," << uPrefix << "_Y,incy);  \n\n";

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+  cublasSetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << uPrefix << "_Y, incy, " << vecYRef
                << ", incy);  \n";

    }

    else {
        lenXY = uPrefix + "_lenXY";
        cocciStream << "+ int " << lenXY << "; \n";
        cocciStream << "+ if(*(trans) == \'N\') " << lenXY << " = n; \n";
        cocciStream << "+ else " << lenXY << " = m; \n\n";

        cocciStream << "+  /* Allocate device memory */  \n";
        cocciStream << "+  cublasAlloc(*(m) * *(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(" << lenXY << ", sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_X);  \n";
        cocciStream << "+  cublasAlloc(" << lenXY << ", sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_Y);  \n\n";
        cocciStream << "+  /* Copy matrix, vectors to device */     \n";
        cocciStream << "+  cublasSetMatrix ( *(m), *(n), sizeType_" << uPrefix
                << ", (void *)" << matARef << ", *(m), (void *) " << uPrefix
                << "_A, *(m));  \n";
        cocciStream << "+  cublasSetVector ( " << lenXY << ",, sizeType_"
                << uPrefix << "," << vecXRef << ", *(incx), " << uPrefix
                << "_X, *(incx));  \n";
        cocciStream << "+  if(*(beta) != 0) cublasSetVector (" << lenXY
                << ", sizeType_" << uPrefix << "," << vecYRef << ", *(incy), "
                << uPrefix << "_Y, *(incy));  \n\n";

        cocciStream << "+  /* CUBLAS call */  \n";
        cocciStream << "+  " << cublasCall
                << "(*(trans),*(m), *(n), *(kl), *(ku), *(alpha)," << uPrefix
                << "_A,*(lda)," << uPrefix << "_X,*(incx),*(beta)," << uPrefix
                << "_Y,*(incy));  \n\n";

        cocciStream << "+  /* Copy result vector back to host */  \n";
        cocciStream << "+  cublasSetVector ( " << lenXY << ", sizeType_"
                << uPrefix << "," << uPrefix << "_Y, *(incy), " << vecYRef
                << ", *(incy));  \n";
    }

    FreeDeviceMemoryB2(cocciStream, uPrefix, true, true, true);
    cocciFptr << cocciStream.str();

}

