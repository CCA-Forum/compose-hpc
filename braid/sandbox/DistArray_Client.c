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
   
  printf("DistArray_Client:getFromDistArray2d_int_Stub(*, 1, 1) \n");
  int res1 = getFromDistArray2d_int_Stub(&distArrayWrapper, 1, 1);
  printf("1. res = %d \n", res1); 

  printf("DistArray_Client:setIntoDistArray2d_int_Stub(*, 1, 1) \n");
  setIntoDistArray2d_int_Stub(&distArrayWrapper, 628, 1, 1);
  
  printf("DistArray_Client:getFromDistArray2d_int_Stub(*, 1, 1) \n");
  int res2 = getFromDistArray2d_int_Stub(&distArrayWrapper, 1, 1);
  printf("2. res = %d \n", res2); 
}
