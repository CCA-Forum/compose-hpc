#include "Blas2Cublas.h"

using namespace std;

void handleSYHERK(ofstream &cocciFptr, bool checkBlasCallType, bool isRowMajor,
        string fname, string uPrefix, SgExprListExp* fArgs) {

    ostringstream cocciStream;
    string matARef = "";
    string matBRef = "";
    string matCRef = "";
    string aType = "";
    string blasCall = fname;
    string cublasCall = "";
    string cuTrans = "";
    string cuUplo = "";
    string cblasUplo = "";
    string cblasTrans = "";

    if (checkBlasCallType) {

        cblasUplo = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(2)->unparseToString();
    }

    else {
        cblasUplo = fArgs->get_traversalSuccessorByIndex(0)->unparseToString();
        cblasTrans = fArgs->get_traversalSuccessorByIndex(1)->unparseToString();
    }

    SgNode* matrixAptr = NULL;
    SgNode* matrixCptr = NULL;

    if (checkBlasCallType) {

        matrixAptr = fArgs->get_traversalSuccessorByIndex(6);
        matrixCptr = fArgs->get_traversalSuccessorByIndex(9);
    } else {
        matrixAptr = fArgs->get_traversalSuccessorByIndex(5);
        matrixCptr = fArgs->get_traversalSuccessorByIndex(8);
    }

    matARef = matrixAptr->unparseToCompleteString();
    matCRef = matrixCptr->unparseToCompleteString();

    if (fname.find("cherk") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCherk";
    } else if (fname.find("zherk") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZherk";
    } else if (fname.find("ssyrk") != string::npos) {
        aType = "float";
        cublasCall = "cublasSsyrk";
    } else if (fname.find("dsyrk") != string::npos) {
        aType = "double";
        cublasCall = "cublasDsyrk";
    } else if (fname.find("csyrk") != string::npos) {
        aType = "cuComplex";
        cublasCall = "cublasCsyrk";
    } else if (fname.find("zsyrk") != string::npos) {
        aType = "cuDoubleComplex";
        cublasCall = "cublasZsyrk";
    }
    cocciStream << "@disable paren@ \n";
    cocciStream << "expression order,uplo,trans;  \n";
    cocciStream << "expression n,k,alpha,a,lda,beta,c,ldc;  \n";
    cocciStream << "@@ \n";
    if (checkBlasCallType)
        cocciStream << "- " << blasCall << "(order,uplo,trans,n,k,alpha,"
                << matARef << ",lda,beta," << matCRef << ",ldc);  \n";
    else
        cocciStream << "- " << blasCall << "(uplo,trans,n,k,alpha," << matARef
                << ",lda,beta," << matCRef << ",ldc);  \n\n";
    cocciStream << "+  /* Allocate device memory */  \n";
    DeclareDevicePtrB3(cocciStream, aType, uPrefix, true, false, true);

    string rA = "";
    string cA = "";
    string dimC = "n";

    if (checkBlasCallType) {

        if (cblasTrans == "CblasTrans") {
            cuTrans = "\'T\'";
            rA = "k";
            cA = "n";
        } else if (cblasTrans == "CblasNoTrans") {
            cuTrans = "\'N\'";
            rA = "n";
            cA = "k";
        } else if (cblasTrans == "CblasConjTrans") {
            cuTrans = "\'C\'";
            rA = "k";
            cA = "n";
        }

        else {
            cuTrans = uPrefix + "_trans";
            rA = uPrefix + "_rA";
            cocciStream << "+ int " << rA << "; \n";
            cA = uPrefix + "_cA";
            cocciStream << "+ int " << cA << "; \n";
            cocciStream << "+ char " << cuTrans << "; \n";
            cocciStream << "+ if(" << cblasTrans << " == CblasTrans) "
                    << cuTrans << " = \'T\'; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasNoTrans) "
                    << cuTrans << " = \'N\'; \n";
            cocciStream << "+ else if(" << cblasTrans << " == CblasConjTrans) "
                    << cuTrans << " = \'C\'; \n\n";
            cocciStream << "+ if(" << cuTrans << " == CblasNoTrans) { " << rA
                    << " = n; " << cA << " = k; } \n";
            cocciStream << "+ else { " << rA << " = k; " << cA
                    << " = n; } \n\n";
        }

        if (cblasUplo == "CblasUpper")
            cuUplo = "\'U\'";
        else if (cblasUplo == "CblasLower")
            cuUplo = "\'L\'";
        else {
            cuUplo = uPrefix + "_uplo";
            cocciStream << "+ char " << cuUplo << "; \n";
            cocciStream << "+ if(" << cblasUplo << " == CblasUpper) " << cuUplo
                    << " = \'U\'; \n";
            cocciStream << "+ else " << cuUplo << " = \'L\'; \n";

        }

        cocciStream << "+  cublasAlloc(n*k, sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(n*n, sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_C);  \n\n";

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+  cublasSetMatrix (" << rA << "," << cA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << rA << ", (void *) " << uPrefix << "_A," << rA << ");  \n\n";
        cocciStream << "+  /* CUBLAS call */  \n";
        RowMajorWarning(cocciStream, isRowMajor);
        cocciStream << "+  " << cublasCall << "(" << cuUplo << "," << cuTrans
                << ",n,k,alpha," << uPrefix << "_A,lda,beta," << uPrefix
                << "_C,ldc);  \n\n";
        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+  cublasSetMatrix( n, n, sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, n, (void *)" << matCRef
                << ", n); \n";

    }

    else {

        rA = uPrefix + "_rA";
        cA = uPrefix + "_cA";

        cocciStream << "+ int " << rA << "; \n";
        cocciStream << "+ int " << cA << "; \n";

        cocciStream << "+ if(*(trans) == \'N\') { " << rA << " = n; " << cA
                << " = k; }\n";
        cocciStream << "+ else { " << rA << " = k; " << cA << " = n; }\n\n";

        cocciStream << "+  cublasAlloc(*(n) * *(k), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_A);  \n";
        cocciStream << "+  cublasAlloc(*(n) * *(n), sizeType_" << uPrefix
                << ", (void**)&" << uPrefix << "_C);  \n\n";

        cocciStream << "+  /* Copy matrices to device */   \n";
        cocciStream << "+  cublasSetMatrix (" << rA << "," << cA
                << ", sizeType_" << uPrefix << ", (void *)" << matARef << ","
                << rA << ", (void *) " << uPrefix << "_A," << rA << "); \n\n";
        cocciStream << "+  /* CUBLAS call */  \n";

        cocciStream << "+  " << cublasCall
                << "(*(uplo),*(trans),*(n),*(k),*(alpha)," << uPrefix
                << "_A,*(lda),*(beta)," << uPrefix << "_C,*(ldc));  \n\n";
        cocciStream << "+  /* Copy result array back to host */ \n";
        cocciStream << "+  cublasSetMatrix( *(n), *(n), sizeType_" << uPrefix
                << ", (void *) " << uPrefix << "_C, *(n), (void *)" << matCRef
                << ", *(n)); \n";
    }

    FreeDeviceMemoryB3(cocciStream, uPrefix, true, false, true);
    cocciFptr << cocciStream.str();

}

