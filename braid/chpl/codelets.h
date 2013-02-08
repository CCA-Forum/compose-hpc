#ifndef included_codelets_h
#define included_codelets_h

#include <sidlType.h>
#include <sidlArray.h>

// Identity function. We use it to cast [anything] -> opaque
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
#define ior_ptr(A) A
#define generic_ptr(A) (void*)A
#define ptr_generic(A) (struct sidl__array*)A
        
#define getOpaqueData(inData) ((void*)inData)
#define isSameOpaqueData(in1, in2) (getOpaqueData(in1) == getOpaqueData(in2))

#define is_null(aPtr)     ((aPtr) == 0)
#define is_not_null(aPtr) ((aPtr) != 0)
#define set_to_null(aPtr) ((*aPtr) = 0)

inline sidl_bool chpl_bool_to_ior_bool(chpl_bool chpl) { return chpl; }

#endif
