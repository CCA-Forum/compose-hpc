#include <stdio.h>
#include "DistArray_Stub.h"
#include "DistArray_Client.h"

void main_server() {

  printf("=========================================== \n");
  printf("   DistArray_Client:Inside main_server() \n");
  printf("=========================================== \n");

  BlockDistArray2d distArrayWrapper;
  
  printf("DistArray_Client:createBlockDistArray2d_int_Stub \n");
  createBlockDistArray2d_int_Stub(&distArrayWrapper, 1, 4, 1, 4, 2, 2);
   
  for (int i = 1; i <= 4; i++) {
    for (int j = 1; j <= 4; j++) {
      int res2 = getFromDistArray2d_int_Stub(&distArrayWrapper, i, j);
      printf("DistArray_Client: array(%d, %d) = %d \n", i, j, res2);
    }
  }

  for (int i = 1; i <= 4; i++) {
    for (int j = 1; j <= 4; j++) {
      printf("DistArray_Client:setIntoDistArray2d_int_Stub(*, %d, %d) \n", i, j);
      setIntoDistArray2d_int_Stub(&distArrayWrapper, 10 * i + j, i, j);
    }
  }
  
  for (int i = 1; i <= 4; i++) {
    for (int j = 1; j <= 4; j++) {
      int res2 = getFromDistArray2d_int_Stub(&distArrayWrapper, i, j);
      printf("DistArray_Client: array(%d, %d) = %d \n", i, j, res2);
    }
  }
}
