      SUBROUTINE icsd_t1_1(d_a,k_a_offset,d_c,k_c_offset)
C     $Id: icsd_t1.F 19699 2010-10-29 17:07:13Z d3y133 $
C     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
C     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
C     i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f
      IMPLICIT NONE
#include"global.fh"
#include"mafdecls.fh"
#include"sym.fh"
#include"errquit.fh"
#include"tce.fh"
      INTEGER d_a, d_c
      INTEGER k_a_offset, k_c_offset
      INTEGER NXTASK, next, nprocs, count
      INTEGER p2b, h1b, p2b_1, h1b_1
      INTEGER dim_common, dima_sort, dima, dimc
      INTEGER k_a, l_a
      EXTERNAL NXTASK
      nprocs = GA_NNODES()
      count = 0
      next = NXTASK(nprocs, 1)
      DO p2b = noab+1,noab+nvab
       DO h1b = 1,noab
        IF (next.eq.count) THEN
         IF ((.not.restricted).or.(int_mb(k_spin+p2b-1)
     1                            +int_mb(k_spin+h1b-1).ne.4)) THEN
          IF (int_mb(k_spin+p2b-1) .eq. int_mb(k_spin+h1b-1)) THEN
           IF (ieor(int_mb(k_sym+p2b-1),int_mb(k_sym+h1b-1)) 
     1         .eq. irrep_f) THEN
            dimc = int_mb(k_range+p2b-1) * int_mb(k_range+h1b-1)
            CALL TCE_RESTRICTED_2(p2b,h1b,p2b_1,h1b_1)
            dim_common = 1
            dima_sort = int_mb(k_range+p2b-1) * int_mb(k_range+h1b-1)
            dima = dim_common * dima_sort
            IF (dima .gt. 0) THEN
             IF (.not.MA_PUSH_GET(mt_dbl,dima,'a',l_a,k_a)) 
     1            CALL ERRQUIT('icsd_t1_1',1,MA_ERR)
             CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,
     1            int_mb(k_a_offset),
     2            (h1b_1 - 1 + (noab+nvab) * (p2b_1 - 1)))
             CALL ADD_HASH_BLOCK(d_c,dbl_mb(k_a),dimc,
     1            int_mb(k_c_offset),
     2            (h1b - 1 + noab * (p2b - noab - 1)))
             IF (.not.MA_POP_STACK(l_a)) 
     1            CALL ERRQUIT('icsd_t1_1',5,MA_ERR)
            END IF
           END IF
          END IF
         END IF
         next = NXTASK(nprocs, 1)
        END IF
        count = count + 1
       END DO
      END DO
      next = NXTASK(-nprocs, 1)
      call GA_SYNC()
      RETURN
      END




