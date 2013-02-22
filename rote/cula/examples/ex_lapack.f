      PROGRAM LAPACK_TEST

        INTEGER M, N, K, STATUS
        PARAMETER ( M = 8192, N = 8192, K = 8192)

        REAL A(M, N)
        REAL TAU(K)

        INTEGER LWORK
        REAL WORK(100)
        INTEGER LDA

        CALL SGEQRF(M, N, A, LDA, TAU, WORK, LWORK, STATUS)

      END

