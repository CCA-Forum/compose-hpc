#ifndef _CHPL_SIDL_ARRAY_H_
#define _CHPL_SIDL_ARRAY_H_

#include <sidlArray.h>
#include <sidl_bool_IOR.h>        
#include <sidl_char_IOR.h>        
#include <sidl_dcomplex_IOR.h>    
#include <sidl_double_IOR.h>      
#include <sidl_fcomplex_IOR.h>    
#include <sidl_float_IOR.h>       
#include <sidl_int_IOR.h>         
#include <sidl_long_IOR.h>        
#include <sidl_opaque_IOR.h>      
#include <sidl_string_IOR.h>      
#include <sidl_BaseInterface_IOR.h>
#include <stdlib.h> 

struct sidl_string__array {
  struct sidl__array   d_metadata;
  char * *d_firstElement;
};


// Chapel-compatible typedef
#define CHAPEL_TYPEDEF(T) \
  typedef struct T _##T; \
  typedef _##T* T;

CHAPEL_TYPEDEF(sidl__array)
CHAPEL_TYPEDEF(sidl_bool__array)        
CHAPEL_TYPEDEF(sidl_char__array)        
CHAPEL_TYPEDEF(sidl_dcomplex__array)    
CHAPEL_TYPEDEF(sidl_double__array)      
CHAPEL_TYPEDEF(sidl_fcomplex__array)    
CHAPEL_TYPEDEF(sidl_float__array)       
CHAPEL_TYPEDEF(sidl_int__array)         
CHAPEL_TYPEDEF(sidl_long__array)        
CHAPEL_TYPEDEF(sidl_opaque__array)      
CHAPEL_TYPEDEF(sidl_string__array)      
CHAPEL_TYPEDEF(sidl_BaseInterface__array)

// Macro definitions for Chapel-generated C code
#define sidlArrayElem1Set(array, ind1, val) \
  sidlArrayElem1(array,ind1) = val

// Identity function. We use it to cast * -> opaque
#define bool_ptr(A) A     
#define char_ptr(A) A     
#define dcomplex_ptr(A) A 
#define double_ptr(A) A   
#define fcomplex_ptr(A) A 
#define float_ptr(A) A    
#define int_ptr(A) A      
#define long_ptr(A) A     
#define opaque_ptr(A) A   
#define string_ptr(A) A   
#define chpl_sidl__array_smartCopy(A) sidl__array_smartCopy((struct sidl__array *)(A))
#define chpl_sidl__array_deleteRef(A) sidl__array_deleteRef((struct sidl__array *)(A))
#define chpl_sidl__array_isColumnOrder(A) sidl__array_isColumnOrder((const struct sidl__array *)(A))
#define chpl_sidl__array_isRowOrder(A) sidl__array_isRowOrder((const struct sidl__array *)(A))
        
#define getOpaqueData(inData) ((void*)inData)
#define isSameOpaqueData(in1, in2) (getOpaqueData(in1) == getOpaqueData(in2))
        
// void* allocateData(int typeSize, int numElements) 
#define allocateData(typeSize, numElements) (chpl_malloc(numElements, typeSize, CHPL_RT_MD_ARRAY_ELEMENTS, 67, "chpl_sidl_array.h"))

// void deallocateData(void* bData) 
#define deallocateData(bData) (chpl_free(bData, 70, "chpl_sidl_array.h"))
        
#endif
