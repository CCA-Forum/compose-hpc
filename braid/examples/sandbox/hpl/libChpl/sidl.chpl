

// -*- chpl -*- 
// This fragment will preprocessed to yield sidl.chpl during compile time.

module sidl {
  // FIXME: autogenerate this from sidl.sidl
  _extern record sidl_BaseInterface {};
// -*- chpl -*- This fragment will be included in sidl.chpl during compile time.
  enum sidl_array_ordering {
    sidl_general_order=0, /* this must be zero (i.e. a false value) */
    sidl_column_major_order=1,
    sidl_row_major_order=2
  };
  enum sidl_array_type {
    sidl_undefined_array = 0,
    /* these values must match values used in F77 & F90 too */
    sidl_bool_array = 1,
    sidl_char_array = 2,
    sidl_dcomplex_array = 3,
    sidl_double_array = 4,
    sidl_fcomplex_array = 5,
    sidl_float_array = 6,
    sidl_int_array = 7,
    sidl_long_array = 8,
    sidl_opaque_array = 9,
    sidl_string_array = 10,
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

  _extern record sidl__array {
    /* int32_t                         *d_lower; */
    /* int32_t                         *d_upper; */
    /* int32_t                         *d_stride; */
    /* const struct sidl__array_vtable *d_vtable; */
    var d_dimen: int(32);
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
   * Please report bugs to <components@llnl.gov>.
   *
   * \authors <pre>
   *
   * Copyright (c) 2011, Lawrence Livermore National Security, LLC.
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
_extern class sidl_bool__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc bool_ptr(inout firstElement: bool): opaque; _extern proc sidl_bool__array_init( inout firstElement: bool, inout sidl_array: sidl_bool__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_bool__array; _extern proc sidl_bool__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_bool__array; proc borrow_bool_Array(inout a: [?dom_a]bool, in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_bool__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(bool, sidl_bool__array, ior); }
_extern class sidl_char__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc char_ptr(inout firstElement: string): opaque; _extern proc sidl_char__array_init( inout firstElement: string, inout sidl_array: sidl_char__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_char__array; _extern proc sidl_char__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_char__array; proc borrow_char_Array(inout a: [?dom_a]string, in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_char__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(string, sidl_char__array, ior); }
_extern class sidl_dcomplex__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc dcomplex_ptr(inout firstElement: complex(128)): opaque; _extern proc sidl_dcomplex__array_init( inout firstElement: complex(128), inout sidl_array: sidl_dcomplex__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_dcomplex__array; _extern proc sidl_dcomplex__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_dcomplex__array; proc borrow_dcomplex_Array(inout a: [?dom_a]complex(128), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_dcomplex__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(complex(128), sidl_dcomplex__array, ior); }
_extern class sidl_double__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc double_ptr(inout firstElement: real(64)): opaque; _extern proc sidl_double__array_init( inout firstElement: real(64), inout sidl_array: sidl_double__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_double__array; _extern proc sidl_double__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_double__array; proc borrow_double_Array(inout a: [?dom_a]real(64), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_double__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(real(64), sidl_double__array, ior); }
_extern class sidl_fcomplex__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc fcomplex_ptr(inout firstElement: complex(64)): opaque; _extern proc sidl_fcomplex__array_init( inout firstElement: complex(64), inout sidl_array: sidl_fcomplex__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_fcomplex__array; _extern proc sidl_fcomplex__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_fcomplex__array; proc borrow_fcomplex_Array(inout a: [?dom_a]complex(64), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_fcomplex__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(complex(64), sidl_fcomplex__array, ior); }
_extern class sidl_float__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc float_ptr(inout firstElement: real(32)): opaque; _extern proc sidl_float__array_init( inout firstElement: real(32), inout sidl_array: sidl_float__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_float__array; _extern proc sidl_float__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_float__array; proc borrow_float_Array(inout a: [?dom_a]real(32), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_float__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(real(32), sidl_float__array, ior); }
_extern class sidl_int__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc int_ptr(inout firstElement: int(32)): opaque; _extern proc sidl_int__array_init( inout firstElement: int(32), inout sidl_array: sidl_int__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_int__array; _extern proc sidl_int__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_int__array; proc borrow_int_Array(inout a: [?dom_a]int(32), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_int__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(int(32), sidl_int__array, ior); }
_extern class sidl_long__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc long_ptr(inout firstElement: int(64)): opaque; _extern proc sidl_long__array_init( inout firstElement: int(64), inout sidl_array: sidl_long__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_long__array; _extern proc sidl_long__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_long__array; proc borrow_long_Array(inout a: [?dom_a]int(64), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_long__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(int(64), sidl_long__array, ior); }
_extern class sidl_opaque__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc opaque_ptr(inout firstElement: opaque): opaque; _extern proc sidl_opaque__array_init( inout firstElement: opaque, inout sidl_array: sidl_opaque__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_opaque__array; _extern proc sidl_opaque__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_opaque__array; proc borrow_opaque_Array(inout a: [?dom_a]opaque, in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_opaque__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(opaque, sidl_opaque__array, ior); }
_extern class sidl_string__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc string_ptr(inout firstElement: string): opaque; _extern proc sidl_string__array_init( inout firstElement: string, inout sidl_array: sidl_string__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_string__array; _extern proc sidl_string__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_string__array; proc borrow_string_Array(inout a: [?dom_a]string, in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_string__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(string, sidl_string__array, ior); }
_extern class sidl_BaseInterface__array { var d_metadata: sidl__array; var d_firstElement: opaque; }; _extern proc BaseInterface_ptr(inout firstElement: int(32)): opaque; _extern proc sidl_BaseInterface__array_init( inout firstElement: int(32), inout sidl_array: sidl_BaseInterface__array, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_BaseInterface__array; _extern proc sidl_BaseInterface__array_borrow( in firstElement: opaque, in dimen: int(32), inout lower: int(32), inout upper: int(32), inout stride: int(32)): sidl_BaseInterface__array; proc borrow_BaseInterface_Array(inout a: [?dom_a]int(32), in firstElement: opaque) { var rank = dom_a.rank; var lus = computeLowerUpperAndStride(a); var lower = lus(0); var upper = lus(1); var stride = lus(2); if (here.id != a.locale.id) { halt( "Non-local access! here = " + here.id + ", a.locale = " + a.locale.id); } var ior = sidl_BaseInterface__array_borrow(firstElement, dom_a.rank, lower[1], upper[1], stride[1]); return new Array(int(32), sidl_BaseInterface__array, ior); }
  // Not a good idea at all: better re-wrap it in a new borrowedArray
  // after we return from an external call.
//  // These variables shall alias with the (struct sidl_array)
//  // d_metadata member of struct sidl_<TYPE>__array
// d_lower :opaque;
// d_upper :opaque;
// d_stride:opaque;
// d_vtable:opaque;
// d_dimen:    int(32);
// d_refcount: int(32);
//  // This pointer shall alias with the d_firstElement pointer of struct sidl_<TYPE>__array
//  int* _data;
//}
  class Array {
    // Actual Chapel defintions
    type ScalarType, IORtype;
    var ior: IORtype;//sidl_TYPE__array;
    //var borrowed: [?dom]ScalarType;
    proc Array(type ScalarType, type IORtype, in ior: IORtype) {
      this.ior = ior;
    }
    /* proc Array(type ScalarType, type IORtype, inout borrow_from: [?dom]ScalarType) { */
    /*   this.borrowed = borrow_from; */
    /* }     */
    /**
     * always returns true, since there is no way to construct an
     * unintialized Array for now.
     */
    proc _not_nil(): bool {
      //writeln(ior.d_metadata);
      return true;
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
      _extern proc sidlArrayDim(inout array: sidl__array): int(32);
      return sidlArrayDim(ior.d_metadata);
    }
    //#define sidlArrayDim(array) (((const struct sidl__array *)(array))->d_dimen)
    /**
     * Macro to return the lower bound on the index for dimension ind of array.
     * A valid index for dimension ind must be greater than or equal to
     * sidlLower(array,ind).
     */
    proc lower(in ind: int(32)): int(32) {
      _extern proc sidlLower(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlLower(ior.d_metadata, ind);
    }
    //#define sidlLower(array,ind) (((const struct sidl__array *)(array))->d_lower[(ind)])
    /**
     * Macro to return the upper bound on the index for dimension ind of array.
     * A valid index for dimension ind must be less than or equal to
     * sidlUpper(array,ind).
     */
    proc upper(in ind: int(32)): int(32) {
      _extern proc sidlUpper(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlUpper(ior.d_metadata, ind);
    }
    //#define sidlUpper(array,ind) (((const struct sidl__array *)(array))->d_upper[(ind)])
    /**
     * Macro to return the number of elements in dimension ind of an array.
     */
    proc length(in ind: int(32)): int(32) {
      _extern proc sidlLength(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlLength(ior.d_metadata, ind);
    }
    //#define sidlLength(array,ind) (sidlUpper((array),(ind)) - sidlLower((array),(ind)) + 1)
    /**
     * Macro to return the stride between elements in a particular dimension.
     * To move from the address of element i to element i + 1 in the dimension
     * ind, add sidlStride(array,ind).
     */
    proc stride(in ind: int(32)): int(32) {
      _extern proc sidlStride(inout array: sidl__array, in ind: int(32)): int(32);
      return sidlStride(ior.d_metadata, ind);
    }
    //#define sidlStride(array,ind) (((const struct sidl__array *)(array))->d_stride[(ind)])
    /**
     * Helper macro for calculating the offset in a particular dimension.
     * This macro makes multiple references to array and ind, so you should
     * not use ++ or -- on arguments to this macro.
     */
    proc arrayDimCalc(in ind: int(32), in v: int(32)): int(32) {
      _extern proc sidlArrayDimCalc(inout array: sidl__array, in ind: int(32), in v: int(32)): int(32);
      return sidlArrayDimCalc(ior.d_metadata, ind, v);
    }
    //#define sidlArrayDimCalc(array, ind, var) (sidlStride(array,ind)*((var) - sidlLower(array,ind)))
    /**
     * Return the address of an element in a one dimensional array.
     * This macro may make multiple references to array and ind1, so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr1(array, ind1)     //  ((array)->d_firstElement + sidlArrayDimCalc(array, 0, ind1))
    /**
     * Macro to return an element of a one dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make multiple references
     * to array and ind1, so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayElem1(array, ind1)     //  (*(sidlArrayAddr1(array,ind1)))
    _extern proc sidlArrayElem1(inout array: sidl__array, in ind: int(32)): ScalarType;
    _extern proc sidlArrayElem1(inout array: sidl__array, in ind: int(32),
                                in val: int(32)): ScalarType;
    proc get(in ind: int(32)): ScalarType { return sidlArrayElem1(ior, ind); }
    proc set(in ind: int(32), val: ScalarType) { sidlArrayElem1Set(ior, ind, val); }
    /**
     * Return the address of an element in a two dimensional array.
     * This macro may make multiple references to array, ind1 & ind2; so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr2(array, ind1, ind2)     //  (sidlArrayAddr1(array, ind1) + sidlArrayDimCalc(array, 1, ind2))
    /**
     * Macro to return an element of a two dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make  multiple
     * references to array, ind1 and ind2; so do not use ++ or -- when using
     * this macro.
     */
    //#define sidlArrayElem2(array, ind1, ind2)     //  (*(sidlArrayAddr2(array, ind1, ind2)))
    /**
     * Return the address of an element in a three dimensional array.
     * This macro may make multiple references to array, ind1, ind2 & ind3; so
     * do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr3(array, ind1, ind2, ind3)     //  (sidlArrayAddr2(array, ind1, ind2) + sidlArrayDimCalc(array, 2, ind3))
    /**
     * Macro to return an element of a three dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side). This macro may make multiple references
     * to array, ind1, ind2 & ind3; so do  not use ++ or -- when using this
     * macro.
     */
    //#define sidlArrayElem3(array, ind1, ind2, ind3)     //  (*(sidlArrayAddr3(array, ind1, ind2, ind3)))
    /**
     * Return the address of an element in a four dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3 &
     * ind4; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr4(array, ind1, ind2, ind3, ind4)     //  (sidlArrayAddr3(array, ind1, ind2, ind3) + sidlArrayDimCalc(array, 3, ind4))
    /**
     * Macro to return an element of a four dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3 & ind4; so do not use ++ or -- when
     * using this macro.
     */
    //#define sidlArrayElem4(array, ind1, ind2, ind3, ind4)     //  (*(sidlArrayAddr4(array, ind1, ind2, ind3, ind4)))
    /**
     * Return the address of an element in a five dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4 & ind5; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5)     //  (sidlArrayAddr4(array, ind1, ind2, ind3, ind4) +     //   sidlArrayDimCalc(array, 4, ind5))
    /**
     * Macro to return an element of a five dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4 & ind5; so do not use ++ or
     * -- when using this macro.
     */
    //#define sidlArrayElem5(array, ind1, ind2, ind3, ind4, ind5)     //  (*(sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5)))
    /**
     * Return the address of an element in a six dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4, ind5 & ind6; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6)     //  (sidlArrayAddr5(array, ind1, ind2, ind3, ind4, ind5) +     //   sidlArrayDimCalc(array, 5, ind6))
    /**
     * Macro to return an element of a six dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4, ind5 & ind6; so do not use
     * ++ or -- when using this macro.
     */
    //#define sidlArrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6)     //  (*(sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6)))
    /**
     * Return the address of an element in a seven dimensional array.
     * This macro may make multiple references to array, ind1, ind2, ind3,
     * ind4, ind5, ind6 & ind7; so do not use ++ or -- when using this macro.
     */
    //#define sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7)     //  (sidlArrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6) +     //   sidlArrayDimCalc(array, 6, ind7))
    /**
     * Macro to return an element of a seven dimensional array as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  This macro may make multiple
     * references to array, ind1, ind2, ind3, ind4, ind5, ind6 & ind7; so do not
     * use ++ or -- when using this macro.
     */
    //#define sidlArrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7)     //  (*(sidlArrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7)))
    /**
     * Macro to return an address of a one dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     */
    //#define RarrayAddr1(array, ind1)     //  ((array)+(ind1))
    /**
     * Macro to return an element of a one dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     */
    //#define RarrayElem1(array, ind1)     //  (*(RarrayAddr1(array, ind1)))
    /**
     * Macro to return an address of a two dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr2(array, ind1, ind2, len1)		    //  ((array)+(ind1)+((ind2)*(len1)))
    /**
     * Macro to return an element of a two dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem2(array, ind1, ind2, len1)		    //  (*(RarrayAddr2(array, ind1, ind2, len1)))
    /**
     * Macro to return an address of a three dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr3(array, ind1, ind2, ind3, len1, len2)	    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2)))
    /**
     * Macro to return an element of a three dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem3(array, ind1, ind2, ind3, len1, len2)	    //  (*(RarrayAddr3(array, ind1, ind2, ind3, len1, len2)))
    /**
     * Macro to return an address of a four dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	    //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3)))
    /**
     * Macro to return an element of a four dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem4(array, ind1, ind2, ind3, ind4, len1, len2, len3)	    //  (*(RarrayAddr4(array, ind1, ind2, ind3, ind4, len1, len2, len3)))
    /**
     * Macro to return an address of a five dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4)     //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)))
    /**
     * Macro to return an element of a five dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4)     //  (*(RarrayAddr5(array, ind1, ind2, ind3, ind4, ind5, len1, len2, len3, len4)))
    /**
     * Macro to return an address of a six dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5)     //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +    //   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)))
    /**
     * Macro to return an element of a six dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5)     //  (*(RarrayAddr6(array, ind1, ind2, ind3, ind4, ind5, ind6, len1, len2, len3, len4, len5)))
    /**
     * Macro to return an address of a seven dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6)     //  ((array)+(ind1)+((ind2)*(len1))+((ind3)*(len1)*(len2))+((ind4)*(len1)*(len2)*(len3))+((ind5)*(len1)*(len2)*(len3)*(len4)) +    //   ((ind6)*(len1)*(len2)*(len3)*(len4)*(len5)) + ((ind7)*(len1)*(len2)*(len3)*(len4)*(len5)*(len6)))
    /**
     * Macro to return an element of a seven dimensional rarray as an LVALUE
     * (i.e. it can appear on the left hand side of an assignment operator or it
     * can appear in a right hand side).  (Rarrays are just native arrays, but
     * they are in column order, so these macros may be useful in C.)
     * @param ind? is the element you wish to reference in dimension ?.
     * @param len? is the length of the dimension ?.
     */
    //#define RarrayElem7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6)     //  (*(RarrayAddr7(array, ind1, ind2, ind3, ind4, ind5, ind6, ind7, len1, len2, len3, len4, len5, len6)))
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
      _extern proc chpl_sidl__array_smartCopy(inout a: sidl__array);
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
      _extern proc chpl_sidl__array_deleteRef(inout a: sidl__array);
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
      _extern proc chpl_sidl__array_isColumnOrder(inout a: sidl__array): bool;
      return chpl_sidl__array_isColumnOrder(ior.d_metadata);
    }
    //sidl_bool
    //sidl__array_isColumnOrder(const struct sidl__array* array);
    /**
     * Return a true value iff the array is a contiguous row-major ordered
     * array.  A NULL array argument causes 0 to be returned.
     */
    proc isRowOrder() {
      _extern proc chpl_sidl__array_isRowOrder(inout a: sidl__array): bool;
      return chpl_sidl__array_isRowOrder(ior.d_metadata);
    }
    //sidl_bool
    //sidl__array_isRowOrder(const struct sidl__array* array);
    /**
     * Return an integer indicating the type of elements held by the
     * array. Zero is returned if array is NULL.
     */
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
// Start: Borrowed Array related items  
// -*- chpl -*- This fragment will be included in sidl.chpl during compile time.
use DSIUtil;
use DefaultRectangular;
///////////////////////////////////////////////////////////////////////////
// The generic borrowed data class
///////////////////////////////////////////////////////////////////////////
_extern proc allocateData(typeSize: int(32), numElements: int(32)): opaque;
_extern proc deallocateData(opData: opaque): opaque;
// dynamic data block class
pragma "data class"
class _borrowedData {
  type eltType;
  var opData: opaque;
  var owner: bool;
  proc ~_borrowedData() {
    // Nothing to free, no data owned!
    // Owner though needs to free allocated data
    if (owner) {
      deallocateData(opData);
    }
  }
  pragma "inline" proc init(opaqueData: opaque) {
    // Just copy over the reference to the data
    this.opData = opaqueData;
    __primitive("ref_borrow", this, "_data", opaqueData);
  }
  pragma "inline" proc this(i: integral) var {
    // rely on chapel compiler to generate lvalue
    return __primitive("array_get", this, i);
  }
}
/**
 * Sample signatures of methods to allocate memory externally.
 *
 * _extern proc allocateData(typeSize: int(32), numElements: int(32)): opaque;
 *
 * _extern proc deallocateData(bData: opaque);
 *
 * _extern proc getBorrowedData(): opaque;
 *
 */
///////////////////////////////////////////////////////////////////////////
// BorrowedDistribution class to create the BorrowedDomain
///////////////////////////////////////////////////////////////////////////
class BorrowedDist: BaseDist {
  proc dsiClone() return this;
  proc dsiAssign(other: this.type) { }
  proc dsiNewRectangularDom(param rank: int, type idxType, param stridable: bool)
    return new BorrowedRectangularDom(rank, idxType, stridable, this);
}
///////////////////////////////////////////////////////////////////////////
// BorrowedRectangular domain to create the borrowed array
// delegates almost all its methods to the standard DefaultRectangularDom
// Main method that has changed:
// - dsiBuildArray()
// - ranges()
///////////////////////////////////////////////////////////////////////////
class BorrowedRectangularDom: BaseRectangularDom {
  param rank : int;
  type idxType;
  param stridable: bool;
  var dist: BorrowedDist;
  // var ranges: rank*range(idxType,BoundedRangeType.bounded,stridable);
  var rectDom: DefaultRectangularDom(rank, idxType, stridable);
  proc linksDistribution() param return false;
  proc dsiLinksDistribution() return false;
  proc BorrowedRectangularDom(param rank, type idxType, param stridable, dist) {
    this.dist = dist;
    this.rectDom = new DefaultRectangularDom(rank, idxType, stridable, defaultDist._value);
  }
  proc initIndices(theLimits ...?k) {
    var myRanges : k * range(idxType, BoundedRangeType.bounded, stridable);
    for param i in 1..k do {
      myRanges(i) = 0..(theLimits(i) - 1);
    }
    dsiSetIndices(myRanges);
  }
  proc ranges(idx) {
    return rectDom.ranges(idx);
  }
  proc dsiClear() { this.rectDom.dsiClear(); }
  // function and iterator versions, also for setIndices
  proc dsiGetIndices() { return this.rectDom.dsiGetIndices(); }
  proc dsiSetIndices(x) { this.rectDom.dsiSetIndices(x); }
  iter these_help(param d: int) {
    for i in this.rectDom.these_help(d) do
      yield i;
  }
  iter these_help(param d: int, block) {
    for i in this.rectDom.these_help(d, block) do
      yield i;
  }
  iter these() {
    for i in this.rectDom.these() do
      yield i;
  }
  iter these(param tag: iterator) where tag == iterator.leader {
    for i in this.rectDom.these(tag) do
      yield i;
  }
  iter these(param tag: iterator, follower) where tag == iterator.follower {
    for i in this.rectDom.these(tag, follower) do
      yield i;
  }
  proc dsiMember(ind: rank*idxType) { return this.rectDom.dsiMember(ind); }
  proc dsiIndexOrder(ind: rank*idxType) { return this.rectDom.dsiIndexOrder(ind); }
  proc dsiDims() { return this.rectDom.dsiDims(); }
  proc dsiDim(d : int) { return this.rectDom.dsiDim(d); }
  // optional, is this necesary? probably not now that
  // homogeneous tuples are implemented as C vectors.
  proc dsiDim(param d : int) { return this.rectDom.dsiDim(d); }
  proc dsiNumIndices { return this.rectDom.dsiNumIndices; }
  proc dsiLow { return this.rectDom.dsiLow; }
  proc dsiHigh { return this.rectDom.dsiHigh; }
  proc dsiAlignedLow { return this.rectDom.dsiAlignedLow; }
  proc dsiAlignedHigh { return this.rectDom.dsiAlignedHigh; }
  proc dsiStride { return this.rectDom.dsiStride; }
  proc dsiAlignment { return this.rectDom.dsiAlignment; }
  proc dsiFirst { return this.rectDom.dsiFirst; }
  proc dsiLast { return this.rectDom.dsiLast; }
  proc dsiBuildArray(type eltType) {
    return new BorrowedRectangularArr(eltType=eltType, rank=rank, idxType=idxType,
                                    stridable=stridable, dom=this);
  }
  proc dsiBuildRectangularDom(param rank: int, type idxType, param stridable: bool,
        ranges: rank*range(idxType, BoundedRangeType.bounded, stridable)) {
    return this.rectDom.dsiBuildRectangularDom(rank, idxType, stridable, ranges);
  }
}
///////////////////////////////////////////////////////////////////////////
// BorrowedRectangular array that can refer to externally allocated memory
// based on DefaultRectangularArr, notable method changes are:
// - this()
// - dsiDestroyData()
// - initialize()
// - dsiReindex()
// - dsiSlice()
// - dsiRankChange()
// - dsiReallocate()
// - dsiLocalSlice()
// - setArrayOrdering()
// - computeForArrayOrdering()
// - setDataOwner()
///////////////////////////////////////////////////////////////////////////
class BorrowedRectangularArr: BaseArr {
  type eltType;
  param rank : int;
  type idxType;
  param stridable: bool;
  var dom : BorrowedRectangularDom(rank=rank, idxType=idxType, stridable=stridable);
  var off: rank*idxType;
  var blk: rank*idxType;
  var str: rank*chpl__signedType(idxType);
  var origin: idxType;
  var factoredOffs: idxType;
  var bData : _borrowedData(eltType);
  var noinit: bool = false;
  var arrayOrdering: sidl_array_ordering = sidl_array_ordering.sidl_row_major_order;
  proc canCopyFromDevice param return true;
  // end class definition here, then defined secondary methods below
  proc setArrayOrdering(in newOrdering: sidl_array_ordering) {
    if (arrayOrdering != newOrdering) {
      arrayOrdering = newOrdering;
      computeForArrayOrdering();
    }
  }
  proc setDataOwner(owner: bool) {
    bData.owner = owner;
  }
  proc computeForArrayOrdering() {
    if (arrayOrdering == sidl_array_ordering.sidl_column_major_order) {
      // Handle column-major ordering blocks
      blk(1) = 1:idxType;
      for param dim in 2..rank do
        blk(dim) = blk(dim - 1) * dom.dsiDim(dim - 1).length;
    } else {
      // Default is assumed to be row-major ordering
      // Compute the block size for row-major ordering
      blk(rank) = 1:idxType;
      for param dim in 1..rank-1 by -1 do
        blk(dim) = blk(dim + 1) * dom.dsiDim(dim + 1).length;
    }
    computeFactoredOffs();
  }
  // can the compiler create this automatically?
  proc dsiGetBaseDom() { return dom; }
  proc dsiDestroyData() {
    if (!bData.owner) {
      // Not the owner, not responsible for deleting the data
      return;
    }
    if dom.dsiNumIndices > 0 {
      pragma "no copy" pragma "no auto destroy" var dr = bData;
      pragma "no copy" pragma "no auto destroy" var dv = __primitive("get ref", dr);
      pragma "no copy" pragma "no auto destroy" var er = __primitive("array_get", dv, 0);
      pragma "no copy" pragma "no auto destroy" var ev = __primitive("get ref", er);
      if (chpl__maybeAutoDestroyed(ev)) {
        for i in 0..dom.dsiNumIndices-1 {
          pragma "no copy" pragma "no auto destroy" var dr = bData;
          pragma "no copy" pragma "no auto destroy" var dv = __primitive("get ref", dr);
          pragma "no copy" pragma "no auto destroy" var er = __primitive("array_get", dv, i);
          pragma "no copy" pragma "no auto destroy" var ev = __primitive("get ref", er);
          chpl__autoDestroy(ev);
        }
      }
    }
    delete bData;
  }
  iter these() var {
    if rank == 1 {
      // This is specialized to avoid overheads of calling dsiAccess()
      if !dom.stridable {
        // This is specialized because the strided version disables the
        // "single loop iterator" optimization
        var first = getDataIndex(dom.dsiLow);
        var second = getDataIndex(dom.dsiLow+dom.ranges(1).stride:idxType);
        var step = (second-first):chpl__signedType(idxType);
        var last = first + (dom.dsiNumIndices-1) * step:idxType;
        for i in first..last by step do
          yield bData(i);
      } else {
        const stride = dom.ranges(1).stride: idxType,
              start = dom.ranges(1).first,
              first = getDataIndex(start),
              second = getDataIndex(start + stride),
              step = (second-first):chpl__signedType(idxType),
              last = first + (dom.ranges(1).length-1) * step:idxType;
        if step > 0 then
          for i in first..last by step do
            yield bData(i);
        else
          for i in last..first by step do
            yield bData(i);
      }
    } else {
      for i in dom do
        yield dsiAccess(i);
    }
  }
  iter these(param tag: iterator) where tag == iterator.leader {
    for follower in dom.these(tag) do
      yield follower;
  }
  iter these(param tag: iterator, follower) var where tag == iterator.follower {
    if debugDefaultDist then
      writeln("*** In array follower code:"); // [\n", this, "]");
    for i in dom.these(tag=iterator.follower, follower) {
      __primitive("noalias pragma");
      yield dsiAccess(i);
    }
  }
  proc computeFactoredOffs() {
    factoredOffs = 0:idxType;
    for param i in 1..rank do {
      factoredOffs = factoredOffs + blk(i) * off(i);
    }
  }
  // change name to setup and call after constructor call sites
  // we want to get rid of all initialize functions everywhere
  proc initialize() {
    if noinit == true then return;
    for param dim in 1..rank {
      off(dim) = dom.dsiDim(dim).alignedLow;
      str(dim) = dom.dsiDim(dim).stride;
    }
    // Compute the block size for row-major ordering
    computeForArrayOrdering();
    // Do not initialize data here, user will explicitly init the data
  }
  proc borrow(opData: opaque) {
    this.bData = new _borrowedData(eltType);
    this.bData.init(opData);
  }
  pragma "inline"
  proc getDataIndex(ind: idxType ...1) where rank == 1 {
    return getDataIndex(ind);
  }
  pragma "inline"
  proc getDataIndex(ind: rank * idxType) {
    var sum = origin;
    if stridable {
      for param i in 1..rank do
        sum += (ind(i) - off(i)) * blk(i) / abs(str(i)):idxType;
    } else {
      for param i in 1..rank do
        sum += ind(i) * blk(i);
      sum -= factoredOffs;
    }
    return sum;
  }
  proc this(ind: idxType ...?numItems) var where rank == numItems {
    var indTuple: numItems * idxType;
    for param i in 1..numItems do {
      indTuple(i) = ind(i);
    }
    return dsiAccess(indTuple);
  }
  proc this(ind: rank*idxType) var {
    return dsiAccess(ind);
  }
  // only need second version because wrapper record can pass a 1-tuple
  pragma "inline"
  proc dsiAccess(ind: idxType ...1) var where rank == 1 {
    return dsiAccess(ind);
  }
  pragma "inline"
  proc dsiAccess(ind : rank*idxType) var {
    if boundsChecking then
      if !dom.dsiMember(ind) then
        halt("array index out of bounds: ", ind);
    var dataInd = getDataIndex(ind);
    //assert(dataInd >= 0);
    //assert(numelm >= 0); // ensure it has been initialized
    //assert(dataInd: uint(64) < numelm: uint(64));
    return bData(dataInd);
  }
  proc dsiReindex(d: DefaultRectangularDom) {
    halt("dsiReindex() not supported for BorrowedRectangularArray");
  }
  proc dsiSlice(d: DefaultRectangularDom) {
    halt("dsiSlice() not supported for BorrowedRectangularArray");
  }
  proc dsiRankChange(d, param newRank: int, param newStridable: bool, args) {
    halt("dsiRankChange() not supported for BorrowedRectangularArray");
  }
  proc dsiReallocate(d: domain) {
    halt("dsiReallocate() not supported for BorrowedRectangularArray");
  }
  proc dsiLocalSlice(ranges) {
    halt("all dsiLocalSlice calls on DefaultRectangulars should be handled in ChapelArray.chpl");
  }
}
proc BorrowedRectangularDom.dsiSerialWrite(f: Writer) {
  f.write("[", dsiDim(1));
  for i in 2..rank do
    f.write(", ", dsiDim(i));
  f.write("]");
}
proc BorrowedRectangularArr.dsiSerialWrite(f: Writer) {
  proc recursiveArrayWriter(in idx: rank*idxType, dim=1, in last=false) {
    var makeStridePositive = if dom.ranges(dim).stride > 0 then 1 else -1;
    if dim == rank {
      var first = true;
      if debugDefaultDist then f.writeln(dom.ranges(dim));
      for j in dom.ranges(dim) by makeStridePositive {
        if first then first = false; else f.write(" ");
        idx(dim) = j;
        f.write(dsiAccess(idx));
      }
    } else {
      for j in dom.ranges(dim) by makeStridePositive {
        var lastIdx = dom.ranges(dim).last;
        idx(dim) = j;
        recursiveArrayWriter(idx, dim=dim+1,
                             last=(last || dim == 1) && (j == lastIdx));
      }
    }
    if !last && dim != 1 then
      f.writeln();
  }
  const zeroTup: rank*idxType;
  recursiveArrayWriter(zeroTup);
}
///////////////////////////////////////////////////////////////////////////
// Utility functions to create borrowed arrays
///////////////////////////////////////////////////////////////////////////
var defaultBorrowedDistr = _newDistribution(new BorrowedDist());
// Define custom copy method for borrowed arrays
proc chpl__initCopy(a: []) where
    a._dom._value.type == BorrowedRectangularDom(a._value.rank, a._value.idxType, a._value.stridable) {
  var b : [a._dom] a.eltType;
  b._value.setArrayOrdering(a._value.arrayOrdering);
  // FIXME: Use reference counting instead of allocating new memory
  if (a._value.bData.owner) {
    // Allocate data and make element-wise copy
    var opData: opaque = allocateData(numBits(a.eltType), a.numElements);
    b._value.borrow(opData);
    b._value.setDataOwner(true);
    [i in a.domain] b(i) = a(i);
  } else {
    // free to borrow data from non-owning array
    b._value.borrow(a._value.bData.opData);
    b._value.setDataOwner(false);
  }
  return b;
}
pragma "inline" proc createBorrowedArray(type arrayIndexType, type arrayElmntType,
        bData: opaque, arrayOrdering: sidl_array_ordering, arraySize: int(32)...?arrayRank) {
  type locDomType = chpl__buildDomainRuntimeType(defaultBorrowedDistr, arrayRank, arrayIndexType, false);
  var locDom: locDomType;
  locDom._value.initIndices((...arraySize));
  type locArrType = chpl__buildArrayRuntimeType(locDom, arrayElmntType);
  var locArr: locArrType;
  locArr._value.setArrayOrdering(arrayOrdering);
  locArr._value.borrow(bData);
  return locArr;
}
pragma "inline" proc getArrayOrdering(sa: sidl.Array) {
  var arrayOrdering: sidl_array_ordering = sidl_array_ordering.sidl_row_major_order;
  if (sa.isColumnOrder()) {
    arrayOrdering = sidl_array_ordering.sidl_column_major_order;
  }
  return arrayOrdering;
}
pragma "inline" proc createBorrowedArray1d(sa: sidl.Array) {
  if (sa.dim() != 1) {
    halt("input array is not of rank 1");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0));
  return bArr;
}
pragma "inline" proc createBorrowedArray2d(sa: sidl.Array) {
  if (sa.dim() != 2) {
    halt("input array is not of rank 2");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1));
  return bArr;
}
pragma "inline" proc createBorrowedArray3d(sa: sidl.Array) {
  if (sa.dim() != 3) {
    halt("input array is not of rank 3");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1), sa.length(2));
  return bArr;
}
pragma "inline" proc createBorrowedArray4d(sa: sidl.Array) {
  if (sa.dim() != 4) {
    halt("input array is not of rank 4");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1), sa.length(2), sa.length(3));
  return bArr;
}
pragma "inline" proc createBorrowedArray5d(sa: sidl.Array) {
  if (sa.dim() != 5) {
    halt("input array is not of rank 5");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1), sa.length(2), sa.length(3),
            sa.length(4));
  return bArr;
}
pragma "inline" proc createBorrowedArray6d(sa: sidl.Array) {
  if (sa.dim() != 6) {
    halt("input array is not of rank 6");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1), sa.length(2), sa.length(3),
            sa.length(4), sa.length(5));
  return bArr;
}
pragma "inline" proc createBorrowedArray7d(sa: sidl.Array) {
  if (sa.dim() != 7) {
    halt("input array is not of rank 7");
  }
  var arrayOrdering = getArrayOrdering(sa);
  var bArr = createBorrowedArray(int(32), sa.ScalarType, sa.first(), arrayOrdering,
            sa.length(0), sa.length(1), sa.length(2), sa.length(3),
            sa.length(4), sa.length(5), sa.length(6));
  return bArr;
}
proc resetBorrowedArray(bArr: BorrowedRectangularArr, bData: opaque,
        arraySize ...?arrayRank) where bArr.rank == arrayRank {
  var bDom = bArr.getBaseDom();
  bDom.initIndices((...arraySize));
  bArr.borrow(bData);
  return bArr;
}
proc isBorrowedArray(in a: [?aDom]): bool {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    return true;
  }
  return false;
}
///////////////////////////////////////////////////////////////////////////
// Start example use of borrowed array
///////////////////////////////////////////////////////////////////////////
//
// _extern proc allocateData(typeSize: int(32), numElements: int(32)): opaque;
//
// type arrayIndexType = int(32);
// type arrayElmntType = real(64);
// var arraySize1d = 10;
//
// var bData1d: opaque;
// local { bData1d = allocateData(numBits(arrayElmntType), arraySize1d); }
//
// var bArr1d = createBorrowedArray(arrayIndexType, arrayElmntType, bData1d,
//       sidl_array_ordering.sidl_row_major_order, arraySize1d);
// [i in 0.. #arraySize1d by 2] { bArr1d(i) = i; }
// [i in 0.. #arraySize1d] { writeln("bArr1d(", i, ") = ", bArr1d(i)); }
//
// var arraySize2di = 3;
// var arraySize2dj = 5;
//
// var bData2d: opaque;
// local { bData2d = allocateData(numBits(arrayElmntType), arraySize2di * arraySize2dj); }
//
// var bArr2d = createBorrowedArray(arrayIndexType, arrayElmntType, bData2d,
//       sidl_array_ordering.sidl_column_major_order, arraySize2di, arraySize2dj);
// [(i, j) in [0.. #arraySize2di, 0.. #arraySize2dj by 2]] { bArr2d(i, j) = (10 * i) + j; }
// [(i, j) in [0.. #arraySize2di, 0.. #arraySize2dj]] { writeln("bArr2d(", i, ", ", j, ") = ", bArr2d(i, j)); }
//
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
// End: Borrowed Array related items  
// Start: Distributed Array related items  
// FIXME Enable the include for darray.inc when we can support gasnet compilation
// # include <darray.inc>
// -*- chpl -*- This fragment will be included in sidl.chpl during compile time.
// TODO Remove all debug prints when done
_extern proc getOpaqueData(inout inData): opaque;
proc performSanityCheck(aDom: domain, varName: string) {
  if (!isRectangularDom(aDom)) {
    halt(varName, ".domain is not rectangular");
  }
  if (aDom._value.stridable) {
    halt(varName, ".domain is stridable. Stridable domains are not supported.");
  }
}
//pragma "inline"
proc getArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    return a._value.arrayOrdering;
  }
  // Default ordering for chapel arrays is row-major
  return sidl_array_ordering.sidl_row_major_order;
}
//pragma "inline"
proc getSidlArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    // Borrowed arrays have inherited their mode
    return a._value.arrayOrdering;
  }
  // Default ordering for rarrays is column-major
  return sidl_array_ordering.sidl_column_major_order;
}
//pragma "inline"
proc ensureLocalArray(inout a:[?aDom], aData: opaque) var
    where isRectangularDom(aDom) {
  param arrayRank = a.rank;
  var arrayOrder = getSidlArrayOrderMode(a);
  var arrayOrderMatch = (getArrayOrderMode(a) == arrayOrder);
  // Create the borrowed domain
  type locDomType = chpl__buildDomainRuntimeType(defaultBorrowedDistr, arrayRank, aDom._value.idxType, false);
  var locDom: locDomType;
  // compute and fill b-array dimension lengths
  var dimRanges: arrayRank * range(aDom._value.idxType, BoundedRangeType.bounded, false);
  if (arrayRank == 1) {
    dimRanges(1) = aDom.low..aDom.high;
  } else {
    for param i in 1..arrayRank do {
      dimRanges(i) = aDom.low(i)..aDom.high(i);
    }
  }
  locDom._value.dsiSetIndices(dimRanges);
  // create the borrowed array in expected sidl mode
  type locArrType = chpl__buildArrayRuntimeType(locDom, a.eltType);
  var locArr: locArrType;
  locArr._value.setArrayOrdering(arrayOrder);
  // borrow data
  if (here.id == aDom.locale.id && (arrayRank == 1 || arrayOrderMatch)) {
    // directly borrow the data for a local array in the correct mode
    var opData: opaque = aData;
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(false);
  } else {
    // make a local copy of the non-local/distributed array in correct
    // order (we expect column-major)
    // self allocate the data, set as owner and then make element-wise copy
    var opData: opaque = allocateData(numBits(locArr.eltType), a.numElements);
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(true);
    [i in aDom] locArr(i) = a(i);
  }
  return locArr;
}
//pragma "inline"
proc syncNonLocalArray(inout src:[], inout target: [?targetDom])
    where isRectangularDom(targetDom) {
  _extern proc isSameOpaqueData(a: opaque, b: opaque): bool;
  var arrayCopyReqd = false;
  if (here.id != targetDom.locale.id) {
    arrayCopyReqd = true;
  } else if (getArrayOrderMode(src) != getArrayOrderMode(target)) {
    arrayCopyReqd = true;
  } else {
    // target is a local array
    var opData1: opaque = getOpaqueData(src(src.domain.low));
    var opData2: opaque = getOpaqueData(target(target.domain.low));
    // If data references are not the same, we need to copy them over
    if (!isSameOpaqueData(opData1, opData2)) {
      arrayCopyReqd = true;
    }
  }
  if (arrayCopyReqd) {
    [i in src.domain] target(i) = src(i);
  }
}
proc checkArraysAreEqual(in srcArray: [], inout destArray: []) {
  [i in srcArray.domain] {
    var srcValue = srcArray(i);
    var destValue = destArray(i);
    if (srcValue != destValue) {
      writeln("ERROR: At index ", i, " expected: ", srcValue, ", but found ", destValue);
    }
  }
}
proc computeLowerUpperAndStride(in srcArray: [?srcDom]) {
  param arrayRank = srcArray.rank;
  var result: [0..2][1..arrayRank] int(32);
  var arrayOrderMode = getArrayOrderMode(srcArray);
  for i in [1..arrayRank] {
    var r: range = srcDom.dim(i);
    result[0][i] = r.low;
    result[1][i] = r.high;
  }
  if (arrayOrderMode == sidl_array_ordering.sidl_column_major_order) {
    var loopStride = 1;
    for i in [1..arrayRank] {
      result[2][i] = loopStride;
      var rs: range = srcDom.dim(i);
      loopStride = loopStride * (rs.high - rs.low + 1);
    }
  } else {
    var loopStride = 1;
    for i in [1..arrayRank] {
      var stride_idx = arrayRank - i + 1;
      result[2][stride_idx] = loopStride;
      var rs: range = srcDom.dim(stride_idx);
      loopStride = loopStride * (rs.high - rs.low + 1);
    }
  }
  return result;
}
// End: Distributed Array related items  
}
