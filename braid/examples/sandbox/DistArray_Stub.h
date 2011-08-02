#ifndef __DIST_ARRAY_STUB_H__
#define __DIST_ARRAY_STUB_H__

// Make forward references of external structs
struct __DistArray_int32_t_2__array_BlockArr_int32_t_2_int32_t_F_BlockArr_int32_t_2_int32_t_F;

// chapel used <class-name>-<eltType>-<rank>-<distArray>
#define BlockDistArrayChpl struct __DistArray_int32_t_2__array_BlockArr_int32_t_2_int32_t_F_BlockArr_int32_t_2_int32_t_F*

// we define our wrapper struct
typedef struct {
  BlockDistArrayChpl actualArray;
} BlockDistArray2d;


void createBlockDistArray2d_int_Stub(
      BlockDistArray2d *wrapperToFill,
      int lo1, int hi1, 
      int lo2, int hi2, 
      int blk1, int blk2);

int getFromDistArray2d_int_Stub(
      BlockDistArray2d* distArray, 
      int id1, int id2);

void setIntoDistArray2d_int_Stub(
      BlockDistArray2d* distArray, 
      int newVal, int id1, int id2);

#endif
