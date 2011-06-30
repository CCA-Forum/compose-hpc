#include "DistArray_Stub.h"
#include <stdio.h>

BlockDistArrayChpl createBlockDistArray2d_int_Skel(
      int lo1, int hi1, int lo2, int hi2, int blk1, int blk2);

int getFromDistArray2d_int_Skel(BlockDistArrayChpl actualArr, int id1, int id2);

void setIntoDistArray2d_int_Skel(BlockDistArrayChpl actualArr, int newVal, int id1, int id2);


void createBlockDistArray2d_int_Stub(
      BlockDistArray2d *wrapperToFill,
      int lo1, int hi1, 
      int lo2, int hi2, 
      int blk1, int blk2) {
  BlockDistArrayChpl T = NULL;
  T = createBlockDistArray2d_int_Skel(lo1, hi1, lo2, hi2, blk1, blk2);
  wrapperToFill->actualArray = T;
}

int getFromDistArray2d_int_Stub(
      BlockDistArray2d* distArray, 
      int id1, int id2) {
  BlockDistArrayChpl actualArr = distArray->actualArray;
  return getFromDistArray2d_int_Skel(actualArr, id1, id2);
}

void setIntoDistArray2d_int_Stub(
      BlockDistArray2d* distArray, 
      int newVal, int id1, int id2) {
  BlockDistArrayChpl actualArr = distArray->actualArray;
  setIntoDistArray2d_int_Skel(actualArr, newVal, id1, id2);
}
