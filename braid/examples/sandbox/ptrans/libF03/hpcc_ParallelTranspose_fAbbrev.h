/* Renaming stub symbols so they don't conflict with F90 bindings */
#ifdef __FORTRAN03__
#define hpcc_ParallelTranspose hpcc_ParallelTranspose_F03
#define hpcc_ParallelTranspose_type hpcc_ParallelTranspose_type_F03
#define hpcc_ParallelTranspose_array hpcc_ParallelTranspose_array_F03
#endif /*__FORTRAN03__*/

