// -*- chpl -*- This fragment will be included in sidl.chpl during compile time.

  extern proc is_not_null(in aRef): bool;
  extern proc is_null(in aRef): bool;
  extern proc generic_ptr(in a):opaque;

  enum sidl_array_ordering {
    sidl_general_order=0, /* this must be zero (i.e. a false value) */
    sidl_column_major_order=1,
    sidl_row_major_order=2
  };

  extern proc sidl__array_type(in ga: opaque): int(32);	       
  enum sidl_array_type {
    sidl_undefined_array = 0,
    /* these values must match values used in F77 & F90 too */
    sidl_bool_array      = 1,
    sidl_char_array      = 2,
    sidl_dcomplex_array  = 3,
    sidl_double_array    = 4,
    sidl_fcomplex_array  = 5,
    sidl_float_array     = 6,
    sidl_int_array       = 7,
    sidl_long_array      = 8,
    sidl_opaque_array    = 9,
    sidl_string_array    = 10,
    sidl_interface_array = 11 /* an array of sidl.BaseInterface's */
  };

  /**
   * The virtual function table for the multi-dimensional arrays of
   * any type.
   */
  /* struct sidl__array_vtable { */

    /*
     * This function should release resources associates with the array
     * passed in.  It is called when the reference count goes to zero.
     */
  /*   void (*d_destroy)(struct sidl__array *); */

    /*
     * If this array controls its own data (i.e. owns the memory), this
     * can simply increment the reference count of the argument and
     * return it.  If the data is borrowed (e.g. a borrowed array), this
     * should make a new array of the same size and copy data from the
     * passed in array to the new array.
     */
  /*   struct sidl__array *(*d_smartcopy)(struct sidl__array *); */

    /*
     * Return the type of the array. The type is an integer value
     * that should match one of the values in enum sidl_array_type.
     */
  /*   int32_t (*d_arraytype)(void); */
  /* }; */

  extern record sidl__array {
    /* int32_t                         *d_lower; */
    /* int32_t                         *d_upper; */
    /* int32_t                         *d_stride; */
    /* const struct sidl__array_vtable *d_vtable; */
    var d_dimen:    int(32);
    var d_refcount: int(32);
  };

  /**
   *
   * Definition of the array data type for chapel
   * http://chapel.cray.com/
   *
   * What we are doing in here is actually wrapping the SIDL Array
   * MACRO(!) API, which is possible because Chapel is eventually being
   * compiled down to C.
   *
   * Please report bugs to <adrian@llnl.gov>.
   *
   * \authors <pre>
   *
   * Copyright (c) 2011-2013, Lawrence Livermore National Security, LLC.
   * Produced at the Lawrence Livermore National Laboratory
   * Written by Adrian Prantl <adrian@llnl.gov>.
   *
   * LLNL-CODE-473891.
   * All rights reserved.
   *
   * This file is part of BRAID. For details, see
   * http://compose-hpc.sourceforge.net/.
   * Please read the COPYRIGHT file for Our Notice and
   * for the BSD License.
   *
   * </pre>
   */

  /**
   * The data structure for multi-dimensional arrays for sidl int.
   * The client may access this with the functions below or using
   * the macros in the header file sidlArray.h.
   */
# define SIDL_ARRAY(C_TYPE, CHAPEL_TYPE)				                \
  extern class sidl_##C_TYPE##__array {					                \
    var d_metadata: sidl__array;					                \
    var d_firstElement: opaque;						                \
  };									                \
  									                \
  /* TODO: this is still just an idead, but it would be nice if these                   \
     functions would be called automatically in get() and set() */                      \
  proc toIOR(in chpl: CHAPEL_TYPE) {        				                \
    extern proc chpl_##C_TYPE##_to_ior_##CTYPE(in chpl:CHAPEL_TYPE);                    \
    return chpl_##C_TYPE##_to_ior_##CTYPE(chpl);			                \
  }									                \
  proc fromIOR(in ior): CHAPEL_TYPE {					                \
    extern proc ior_##C_TYPE##_to_chpl_##CTYPE(in ior): CHAPEL_TYPE;    		\
    return ior_##C_TYPE##_to_chpl_##CTYPE(ior);				                \
  }									                \
                                                                                        \
  extern proc C_TYPE##_ptr(inout firstElement: CHAPEL_TYPE): opaque;			\
  extern proc sidl_##C_TYPE##__array_init(                                              \
                   inout firstElement: CHAPEL_TYPE,                                     \
                   inout sidl_array: sidl_##C_TYPE##__array,				\
                   in dimen: int(32),							\
                   inout lower: int(32),						\
                   inout upper: int(32),						\
                   inout stride: int(32)): sidl_##C_TYPE##__array;			\
  extern proc sidl_##C_TYPE##__array_borrow(						\
                   in firstElement: opaque,						\
                   in dimen: int(32),							\
                   inout lower: int(32),						\
                   inout upper: int(32),						\
                   inout stride: int(32)): sidl_##C_TYPE##__array;			\
                                                                                        \
  /**											\
   * Wrap an existing Chapel array inside of a new SIDL array. The initial		\
   * content it determined by the data being borrowed. 	         			\
   *											\
   * A word of WARNING: An array borrowed from Chapel will only work			\
   * until the function calling sidl_*_array_borrow() returns. If you			\
   * want to retain the array beyond that point you will need to run			\
   * smartCopy().									\
   */											\
  proc borrow_##C_TYPE##_array(inout a: [?dom_a]CHAPEL_TYPE, in firstElement: opaque) {	\
    var rank = dom_a.rank: int(32);					\
    var lus = computeLowerUpperAndStride(a);				\
    var lower = lus(0): int(32);					\
    var upper = lus(1): int(32);					\
    var stride = lus(2): int(32);					\
    if (here.id != a.locale.id) {					\
      halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); \
    }									\
    									\
    var ior = sidl_##C_TYPE##__array_borrow(firstElement,		\
					    rank,			\
					    lower[1],			\
					    upper[1],			\
					    stride[1]);			\
    return new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior);		\
  }									\
									\
  /**									\
   * wrap a raw SIDL IOR array inside of a new Chapel object            \
   */									\
  export wrap_##C_TYPE##_array						\
  proc wrap_##C_TYPE##_array(in ior: sidl_##C_TYPE##__array) {		\
    return new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior);		\
  }									\
									\
  /* Various extern declarations */                                     \
  extern proc sidl_##C_TYPE##__array_create1d(in len: int(32))		\
    : sidl_##C_TYPE##__array;						\
									\
  extern proc sidl_##C_TYPE##__array_create2dCol(in m: int(32), in n: int(32)) \
    : sidl_##C_TYPE##__array;						\
									\
  extern proc sidl_##C_TYPE##__array_cast(in ga: opaque)		\
    : sidl_##C_TYPE##__array;						\
									\
  module C_TYPE##_array {						\
    /**
     * Create a dense one-dimensional vector of doubles with a lower
     * index of 0 and an upper index of len-1. This array owns and manages
     * its data.
     * This function does not initialize the contents of the array.
     */									\
    proc create1d(in len: int(32)) {					\
      var ior = sidl_##C_TYPE##__array_create1d(len);			\
      var sidlArray = new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior); \
      return (sidlArray, createBorrowedArray1d(sidlArray));		\
    }									\
									\
    /**
     * Create a dense two-dimensional array of doubles with a lower
     * indices of (0,0) and an upper indices of (m-1,n-1). The array is
     * stored in column-major order, and it owns and manages its data.
     * This function does not initialize the contents of the array.
     */						                        \
    proc create2dCol(in m: int(32), in n: int(32)) {			\
      var ior = sidl_##C_TYPE##__array_create2dCol(m, n);		\
      var sidlArray = new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior); \
      return (sidlArray, createBorrowedArray2d(sidlArray));		\
    }									\
									\
    /**
     * Create a dense two-dimensional array of doubles with a lower
     * indices of (0,0) and an upper indices of (m-1,n-1). The array is
     * stored in row-major order, and it owns and manages its data.
     * This function does not initialize the contents of the array.
     */						                        \
    proc create2dRow(in m: int(32), in n: int(32)) {			\
      var ior = sidl_##C_TYPE##__array_create2dRow(m, n);		\
      var sidlArray = new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior); \
      return (sidlArray, createBorrowedArray2d(sidlArray));		\
    }									\
									\
    /** create a borrowed array from a sidl.Array */			\
    proc borrow(in sidlArray) {						\
      select sidlArray.ior.d_metadata.d_dimen {				\
	when 1 do return (sidlArray, createBorrowedArray1d(sidlArray)); \
	when 2 do return (sidlArray, createBorrowedArray2d(sidlArray)); \
	when 3 do return (sidlArray, createBorrowedArray3d(sidlArray)); \
	when 4 do return (sidlArray, createBorrowedArray4d(sidlArray)); \
	when 5 do return (sidlArray, createBorrowedArray5d(sidlArray)); \
	when 6 do return (sidlArray, createBorrowedArray6d(sidlArray)); \
	when 7 do return (sidlArray, createBorrowedArray7d(sidlArray)); \
	otherwise return nil;						\
      } 								\
    }	      							        \
									\
    /** cast a generic array to a specific array */			\
    proc cast(in generic_array:opaque) {				\
      if (sidl__array_type(generic_array) !=				\
	  sidl_array_type.sidl_##C_TYPE##_array) then {			\
	return nil;							\
      }									\
      var ior = sidl_##C_TYPE##__array_cast(generic_array);		\
      if is_not_null(ior) then						\
	return new Array(CHAPEL_TYPE, sidl_##C_TYPE##__array, ior);	\
      else return nil;							\
    }	      							        \
  }

SIDL_ARRAY(bool,     bool)
SIDL_ARRAY(char,     string)
SIDL_ARRAY(dcomplex, complex(128))
SIDL_ARRAY(double,   real(64))
SIDL_ARRAY(fcomplex, complex(64))
SIDL_ARRAY(float,    real(32))
SIDL_ARRAY(int,      int(32))
SIDL_ARRAY(long,     int(64))
SIDL_ARRAY(opaque,   opaque)
SIDL_ARRAY(string,   string)
SIDL_ARRAY(interface, opaque)


  class Array {
    // Actual Chapel definitions
    type ScalarType, IORtype;

    /** IOR representation of the array */
    var ior: IORtype; /*sidl_TYPE__array;*/
    /** IOR generic array<> representation of the array */
    var generic: opaque; 

    proc Array(type ScalarType, type IORtype, in ior: IORtype) {
      this.ior = ior;

      extern proc generic_array(ior): sidl__array;
      /* Generic array:

	 We need to use opaque as the Chapel representation because
	 otherwise Chapel would start copying arguments and thus mess
	 with the invariant that genarr.d_metadata == genarr */
      extern proc ior_ptr(ior): opaque;
      this.generic = ior_ptr(ior);
    }

    /**
     * Return true iff the wrapped SIDL array is not NULL.
     */
    proc is_not_nil(): bool {
      return is_not_null(this.generic);
    }

    /**
     * Return true iff the wrapped SIDL array is NULL.
     */
    proc is_nil(): bool {
      return is_null(this.generic);
    }

    /**
     * The relation operators available in the built-in quantifier operators.
     */
    /* #define RELATION_OP_EQUAL         0 */
    /* #define RELATION_OP_NOT_EQUAL     1 */
    /* #define RELATION_OP_LESS_THAN     2 */
    /* #define RELATION_OP_LESS_EQUAL    3 */
    /* #define RELATION_OP_GREATER_THAN  4 */
    /* #define RELATION_OP_GREATER_EQUAL 5 */

    /**
     * Return pointer to first element of the underlying array
     */
    proc first(): opaque {
      return ior.d_firstElement;
    }

    /**
     * Return the dimension of the array.
     */
    proc dim(): int(32) {
      extern proc sidlArrayDim(inout array: sidl__array): int(32);
      return sidlArrayDim(ior.d_metadata);
    }
    //#define sidlArrayDim(array) (((const struct sidl__array *)(array))->d_dimen)


    /**
     * Macro to return the lower bound on the index for dimension ind of array.
     * A valid index for dimension ind must be greater than or equal to
     * sidlLower(array,ind).
     */

    proc lower(in ind: int(32)): int(32) {
      extern proc sidlLower(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlLower(ior.d_metadata, ind);
    }
    //#define sidlLower(array,ind) (((const struct sidl__array *)(array))->d_lower[(ind)])

    /**
     * Macro to return the upper bound on the index for dimension ind of array.
     * A valid index for dimension ind must be less than or equal to
     * sidlUpper(array,ind).
     */
    proc upper(in ind: int(32)): int(32) {
      extern proc sidlUpper(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlUpper(ior.d_metadata, ind);
    }
    //#define sidlUpper(array,ind) (((const struct sidl__array *)(array))->d_upper[(ind)])

    /**
     * Macro to return the number of elements in dimension ind of an array.
     */
    proc length(in ind: int(32)): int(32) {
      extern proc sidlLength(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlLength(ior.d_metadata, ind);
    }
    //#define sidlLength(array,ind) (sidlUpper((array),(ind)) - sidlLower((array),(ind)) + 1)

    /**
     * Macro to return the stride between elements in a particular dimension.
     * To move from the address of element i to element i + 1 in the dimension
     * ind, add sidlStride(array,ind).
     */
    proc stride(in ind: int(32)): int(32) {
      extern proc sidlStride(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlStride(ior.d_metadata, ind);
    }
    //#define sidlStride(array,ind) (((const struct sidl__array *)(array))->d_stride[(ind)])

    /**
     * Helper macro for calculating the offset in a particular dimension.
     * This macro makes multiple references to array and ind, so you should
     * not use ++ or -- on arguments to this macro.
     */
    proc arrayDimCalc(in ind: int(32), in v: int(32)): int(32) {
      extern proc sidlArrayDimCalc(inout array: sidl__array, in ind: int(32), in v: int(32)): int(32);
      return sidlArrayDimCalc(ior.d_metadata, ind, v);
    }
    //#define sidlArrayDimCalc(array, ind, var) (sidlStride(array,ind)*((var) - sidlLower(array,ind)))


    /**
     * Return the address of an element in a one dimensional array.
     * This macro may make multiple references to array and ind1, so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr1(array, ind1) \
    //  ((array)->d_firstElement + sidlArrayDimCalc(array, 0, ind1))

    /**
     * Macro to return an element of a one dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make multiple references
     * to array and ind1, so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayElem1(array, ind1) \
    //  (*(sidlArrayAddr1(array,ind1)))
    extern proc sidlArrayElem1(inout array: IORtype, 
			       in ind: int(32)): ScalarType;

    proc get(in ind: int(32)): ScalarType { 
      return /*fromIOR*/(sidlArrayElem1(ior, ind)); 
    }

    proc set(in ind: int(32), val: ScalarType) { 
       extern proc sidlArrayElem1Set(inout array: IORtype, in ind: int(32), val:ScalarType);
       sidlArrayElem1Set(ior, ind, /*toIOR*/(val)); 
    }

    /**
     * Return the address of an element in a two dimensional array.
     * This macro may make multiple references to array, ind1 & ind2; so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr2(array, ind1, ind2) \
    //  (sidlArrayAddr1(array, ind1) + sidlArrayDimCalc(array, 1, ind2))

    /**
     * Macro to return an element of a two dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make  multiple
     * references to array, ind1 and ind2; so do not use ++ or -- when using
     * this macro.
     */
    //#define sidlArrayElem2(array, ind1, ind2) \
    //  (*(sidlArrayAddr2(array, ind1, ind2)))


    /**
     * Return the address of an element in a three dimensional array.
     * This macro may make multiple references to array, ind1, ind2 & ind3; so
     * do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr3(array, ind1, ind2, ind3) \
    //  (sidlArrayAddr2(array, ind1, ind2) + sidlArrayDimCalc(array, 2, ind3))

    /**
     * Macro to return an element of a three dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make multiple references
     * to array, ind1, ind2 & ind3; so do  not use ++ or -- when using this
     * macro.
     */
    //#define sidlArrayElem3(array, ind1, ind2, ind3) \
    //  (*(sidlArrayAddr3(array, ind1, ind2, ind3)))


    /**
     * Return the address of an element in a four dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3 &
     * ind4; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr4(array, ind1, ind2, ind3, ind4) \
    //  (sidlArrayAddr3(array, ind1, ind2, ind3) + sidlArrayDimCalc(array, 3, ind4))

    /**
     * Macro to return an element of a four dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3 & ind4; so do not use ++ or -- when
     * using this macro.
     */
    //#define sidlArrayElem4(array, ind1, ind2, ind3, ind4) \
    //  (*(sidlArrayAddr4(array, ind1, ind2, ind3, ind4)))

    /**
     * Return the address of an element in a five dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4 & ind5; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5) \
    //  (sidlArrayAddr4(array, ind1, ind2, ind3, ind4) + \
    //   sidlArrayDimCalc(array, 4, ind5))

    /**
     * Macro to return an element of a five dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4 & ind5; so do not use ++ or
     * -- when using this macro.
     */
    //#define sidlArrayElem5(array, ind1, ind2, ind3, ind4, ind5) \
    //  (*(sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5)))

    /**
     * Return the address of an element in a six dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4, ind5 & ind6; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6) \
    //  (sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5) + \
    //   sidlArrayDimCalc(array, 5, ind6))

    /**
     * Macro to return an element of a six dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4, ind5 & ind6; so do not use
     * ++ or -- when using this macro.
     */
    //#define sidlArrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6) \
    //  (*(sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6)))

    /**
     * Return the address of an element in a seven dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4, ind5, ind6 & ind7; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7) \
    //  (sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6) + \
    //   sidlArrayDimCalc(array, 6, ind7))

    /**
     * Macro to return an element of a seven dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4, ind5, ind6 & ind7; so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7) \
    //  (*(sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7)))

    /**
     * Macro to return an address of a one dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     */
    //#define RarrayAddr1(array, ind1) \
    //  ((array)+(ind1))

    /**
     * Macro to return an element of a one dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     */
    //#define RarrayElem1(array, ind1) \
    //  (*(RarrayAddr1(array, ind1)))

    /**
     * Macro to return an address of a two dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr2(array, ind1, ind2, len1)		\
    //  ((array)+(ind1)+((ind2)*(len1)))

    /**
     * Macro to return an element of a two dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem2(array, ind1, ind2, len1)		\
    //  (*(RarrayAddr2(array, ind1, ind2, len1)))

    /**
     * Macro to return an address of a three dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr3(array, ind1, ind2, ind3, len1, len2)	\
    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2)))

    /**
     * Macro to return an element of a three dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem3(array, ind1, ind2, ind3, len1, len2)	\
    //  (*(RarrayAddr3(array, ind1, ind2, ind3, len1, len2)))

    /**
     * Macro to return an address of a four dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	\
    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3)))

    /**
     * Macro to return an element of a four dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	\
    //  (*(RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)))

    /**
     * Macro to return an address of a five dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4) \
    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)))

    /**
     * Macro to return an element of a five dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4) \
    //  (*(RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4)))

    /**
     * Macro to return an address of a six dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5) \
    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +\
    //   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)))

    /**
     * Macro to return an element of a six dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5) \
    //  (*(RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5)))

    /**
     * Macro to return an address of a seven dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6) \
    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +\
    //   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)) + ((ind7)*(len1)*(len2)*(len3)*(len4)*(len5)*(len6)))

    /**
     * Macro to return an element of a seven dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6) \
    //  (*(RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6)))


    /**
     * Increment the arrays internal reference count by one. To make a
     * persistent copy (i.e. that lives longer than the current method
     * call) use smartCopy.
     */
    //void
    //sidl__array_addRef(struct sidl__array* array);

    /**
     * If array is borrowed, allocate a new self-sufficient array and copy
     * the borrowed array into the new array; otherwise, increment the
     * reference count and return the array passed in. Use this whenever
     * you want to make a copy of a method argument because arrays passed
     * into methods aren't guaranteed to exist after the method call.
     */
    proc smartCopy() {
      extern proc chpl_sidl__array_smartCopy(inout a: sidl__array);
      chpl_sidl__array_smartCopy(ior.d_metadata);
    }
    //struct sidl__array *
    //sidl__array_smartCopy(struct sidl__array *array);

    /**
     * Decrement the array's internal reference count by one. If the reference
     * count goes to zero, destroy the array.
     * Return true iff the array is destroyed
     */
    proc deleteRef() {
      extern proc chpl_sidl__array_deleteRef(inout a: sidl__array);
      chpl_sidl__array_deleteRef(ior.d_metadata);
    }
    //void
    //sidl__array_deleteRef(struct sidl__array* array);

    /**
     * Return the dimension of array. If the array pointer is NULL,
     * zero is returned.
     */
    //int32_t
    //sidl__array_dimen(const struct sidl__array* array);

    /**
     * Return the lower index bound on dimension ind. If ind is not a valid
     * dimension, zero is returned. The valid range for ind is 0 to dimen-1.
     */
    //int32_t
    //sidl__array_lower(const struct sidl__array* array,
    //                  const int32_t ind);

    /**
     * Return the upper index bound on dimension ind. If ind is not a valid
     * dimension, negative one is returned. The valid range for ind is 0 to
     * dimen-1.
     */
    //int32_t
    //sidl__array_upper(const struct sidl__array* array,
    //                  const int32_t ind);

    /**
     * Return the number of element in dimension ind. If ind is not a valid
     * dimension, negative one is returned. The valid range for ind is 0 to
     * dimen-1.
     */
    //int32_t
    //sidl__array_length(const struct sidl__array* array,
    //                   const int32_t ind);

    /**
     * Return the stride of dimension ind. If ind is not a valid
     * dimension, zero is returned. The valid range for ind is 0 to
     * dimen-1.
     */
    //int32_t
    //sidl__array_stride(const struct sidl__array* array,
    //                   const int32_t ind);

    /**
     * Return a true value iff the array is a contiguous column-major ordered
     * array.  A NULL array argument causes 0 to be returned.
     */
    proc isColumnOrder() {
      extern proc chpl_sidl__array_isColumnOrder(inout a: sidl__array): bool;
      return chpl_sidl__array_isColumnOrder(ior.d_metadata);
    }
    //sidl_bool
    //sidl__array_isColumnOrder(const struct sidl__array* array);

    /**
     * Return a true value iff the array is a contiguous row-major ordered
     * array.  A NULL array argument causes 0 to be returned.
     */
    proc isRowOrder() {
      extern proc chpl_sidl__array_isRowOrder(inout a: sidl__array): bool;
      return chpl_sidl__array_isRowOrder(ior.d_metadata);
    }
    //sidl_bool
    //sidl__array_isRowOrder(const struct sidl__array* array);

    /**
     * Return an integer indicating the type of elements held by the
     * array. Zero is returned if array is NULL.
     */
    proc arrayType(): sidl_array_type {
      return sidl__array_type(this.generic) :sidl_array_type ;
    }

    //int32_t
    //sidl__array_type(const struct sidl__array* array);

    /**
     * The following two functions are used for low level array reference
     * count debugging. They are not intended for Babel end-users.
     */
    //void
    //sidl__array_add(struct sidl__array * const array);

    //void
    //sidl__array_remove(struct sidl__array * const array);

    }
