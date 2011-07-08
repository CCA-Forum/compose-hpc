
#include "hplsupport.h"
#include "hplsupport_HPL_cClient.h"

#define DEBUG 0

void transposeHelperNative(int32_t aData, int32_t cData, double beta, int i, int j, int hereId) {

  sidl_BaseInterface ex;

  // printf(" locale id = %d, aData = %p, cData = %p \n", hereId, aData, cData);

  hplsupport_BlockCyclicDistArray2dDouble a = hplsupport_BlockCyclicDistArray2dDouble__create(&ex);
  hplsupport_BlockCyclicDistArray2dDouble_initData(a, &aData, &ex);

  hplsupport_BlockCyclicDistArray2dDouble c = hplsupport_BlockCyclicDistArray2dDouble__create(&ex);
  hplsupport_BlockCyclicDistArray2dDouble_initData(c, &cData, &ex);

  // C[i,j] = beta * C[i,j]  +  A[j,i];

  double a_ji = hplsupport_BlockCyclicDistArray2dDouble_get(a, j, i, &ex);
  double c_ij = hplsupport_BlockCyclicDistArray2dDouble_get(c, i, j, &ex);

  double new_val = beta * c_ij  +  a_ji;

  hplsupport_BlockCyclicDistArray2dDouble_set(c, new_val, i, j, &ex);
}
