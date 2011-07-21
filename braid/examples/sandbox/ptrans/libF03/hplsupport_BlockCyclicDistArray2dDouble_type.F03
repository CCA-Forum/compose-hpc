! 
! File:          hplsupport_BlockCyclicDistArray2dDouble_type.F03
! Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 7138  trunk)
! Description:   Client-side module for hplsupport.BlockCyclicDistArray2dDouble
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h"

! 
! This file contains a FORTRAN 2003 derived type for the
! sidl type hplsupport.BlockCyclicDistArray2dDouble.
! 

module hplsupport_BlockCyclicDistArray2dDouble_type
  use sidl
  use, intrinsic :: iso_c_binding

  ! This is a Fortran type that is supposed to be interoperable
  ! with the corresponding C entry point vector.
  type, bind(c) :: hplsupport_BlockCyclicDistArray2dDouble_epv_t
    type(c_funptr) :: f__cast
    type(c_funptr) :: f__delete
    type(c_funptr) :: f__exec
    type(c_funptr) :: f__getURL
    type(c_funptr) :: f__raddRef
    type(c_funptr) :: f__isRemote
    type(c_funptr) :: f__set_hooks
    type(c_funptr) :: f__set_contracts
    type(c_funptr) :: f__dump_stats
    type(c_funptr) :: f__ctor
    type(c_funptr) :: f__ctor2
    type(c_funptr) :: f__dtor
    type(c_funptr) :: f__load
    ! Methods introduced in sidl.BaseInterface-v0.9.17
    type(c_funptr) :: f_addRef
    type(c_funptr) :: f_deleteRef
    type(c_funptr) :: f_isSame
    type(c_funptr) :: f_isType
    type(c_funptr) :: f_getClassInfo
    ! Methods introduced in sidl.BaseClass-v0.9.17
    ! Methods introduced in hplsupport.BlockCyclicDistArray2dDouble-v0.1
    type(c_funptr) :: f_initData
    type(c_funptr) :: f_get
    type(c_funptr) :: f_set
  end type

  ! This is a Fortran type that is supposed to be interoperable
  ! with the corresponding C post-hook entry point vector.
  type, bind(c) :: hplsupport_BlockCyclicDistArray2dDouble_post_epv_t
    type(c_funptr) :: f_initData_post
    type(c_funptr) :: f_get_post
    type(c_funptr) :: f_set_post
  end type

  ! This is a Fortran type that is supposed to be interoperable
  ! with the corresponding C pre-hook entry point vector.
  type, bind(c) :: hplsupport_BlockCyclicDistArray2dDouble_pre_epv_t
    type(c_funptr) :: f_initData_pre
    type(c_funptr) :: f_get_pre
    type(c_funptr) :: f_set_pre
  end type


  type hplsupport_BlockCyclicDistArray2dDouble_t
    type(c_ptr) :: d_ior = c_null_ptr
    type(hplsupport_BlockCyclicDistArray2dDouble_epv_t), pointer :: d_epv =>   &
      null()
  end type hplsupport_BlockCyclicDistArray2dDouble_t

  type hplsupport_BlockCyclicDistArray2dDouble_1d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_1d

  type hplsupport_BlockCyclicDistArray2dDouble_2d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_2d

  type hplsupport_BlockCyclicDistArray2dDouble_3d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_3d

  type hplsupport_BlockCyclicDistArray2dDouble_4d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_4d

  type hplsupport_BlockCyclicDistArray2dDouble_5d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_5d

  type hplsupport_BlockCyclicDistArray2dDouble_6d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_6d

  type hplsupport_BlockCyclicDistArray2dDouble_7d
    type(c_ptr) :: d_array = c_null_ptr
  end type hplsupport_BlockCyclicDistArray2dDouble_7d


  private :: cache_epv_s
  interface cache_epv
    module procedure cache_epv_s
  end interface cache_epv

contains

    subroutine cache_epv_s(self)
      use, intrinsic :: iso_c_binding
      implicit none
      class(hplsupport_BlockCyclicDistArray2dDouble_t) :: self
      type(c_ptr) :: cptr = c_null_ptr
      interface
        type(c_ptr) function get_epv_from_ptr(ior) bind(c,                     &
          name="hplsupport_BlockCyclicDistArray2dDouble_getEPV")
          use iso_c_binding
          type(c_ptr), value :: ior
        end function get_epv_from_ptr
      end interface
      self%d_epv => null()
      if(c_associated(self%d_ior)) then
        cptr = get_epv_from_ptr(self%d_ior)
        if(c_associated(cptr)) then
          call c_f_pointer(cptr, self%d_epv)
        endif
      endif
    end subroutine cache_epv_s

end module hplsupport_BlockCyclicDistArray2dDouble_type

#include "hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h"

module hplsupport_BlockCyclicDistArray2dDouble_array
  use sidl
  use hplsupport_BlockCyclicDistArray2dDouble_type
  use sidl_array_type
  use, intrinsic :: iso_c_binding

  private :: &
    copy1_p, &
    copy2_p, &
    copy3_p, &
    copy4_p, &
    copy5_p, &
    copy6_p, &
    copy7_p

  interface copy
    module procedure &
      copy1_p, &
      copy2_p, &
      copy3_p, &
      copy4_p, &
      copy5_p, &
      copy6_p, &
      copy7_p
  end interface

  private :: &
    createCol1_p, &
    createCol2_p, &
    createCol3_p, &
    createCol4_p, &
    createCol5_p, &
    createCol6_p, &
    createCol7_p

  interface createCol
    module procedure &
      createCol1_p, &
      createCol2_p, &
      createCol3_p, &
      createCol4_p, &
      createCol5_p, &
      createCol6_p, &
      createCol7_p
  end interface

  private :: &
    createRow1_p, &
    createRow2_p, &
    createRow3_p, &
    createRow4_p, &
    createRow5_p, &
    createRow6_p, &
    createRow7_p

  interface createRow
    module procedure &
      createRow1_p, &
      createRow2_p, &
      createRow3_p, &
      createRow4_p, &
      createRow5_p, &
      createRow6_p, &
      createRow7_p
  end interface

  private :: &
    ensure1_p, &
    ensure2_p, &
    ensure3_p, &
    ensure4_p, &
    ensure5_p, &
    ensure6_p, &
    ensure7_p

  interface ensure
    module procedure &
      ensure1_p, &
      ensure2_p, &
      ensure3_p, &
      ensure4_p, &
      ensure5_p, &
      ensure6_p, &
      ensure7_p
  end interface

  private :: &
    addRef1_p, &
    addRef2_p, &
    addRef3_p, &
    addRef4_p, &
    addRef5_p, &
    addRef6_p, &
    addRef7_p

  interface addRef
    module procedure &
      addRef1_p, &
      addRef2_p, &
      addRef3_p, &
      addRef4_p, &
      addRef5_p, &
      addRef6_p, &
      addRef7_p
  end interface

  private :: &
    deleteRef1_p, &
    deleteRef2_p, &
    deleteRef3_p, &
    deleteRef4_p, &
    deleteRef5_p, &
    deleteRef6_p, &
    deleteRef7_p

  interface deleteRef
    module procedure &
      deleteRef1_p, &
      deleteRef2_p, &
      deleteRef3_p, &
      deleteRef4_p, &
      deleteRef5_p, &
      deleteRef6_p, &
      deleteRef7_p
  end interface

  private :: &
    dimen1_p, &
    dimen2_p, &
    dimen3_p, &
    dimen4_p, &
    dimen5_p, &
    dimen6_p, &
    dimen7_p

  interface dimen
    module procedure &
      dimen1_p, &
      dimen2_p, &
      dimen3_p, &
      dimen4_p, &
      dimen5_p, &
      dimen6_p, &
      dimen7_p
  end interface

  private :: &
    isColumnOrder1_p, &
    isColumnOrder2_p, &
    isColumnOrder3_p, &
    isColumnOrder4_p, &
    isColumnOrder5_p, &
    isColumnOrder6_p, &
    isColumnOrder7_p

  interface isColumnOrder
    module procedure &
      isColumnOrder1_p, &
      isColumnOrder2_p, &
      isColumnOrder3_p, &
      isColumnOrder4_p, &
      isColumnOrder5_p, &
      isColumnOrder6_p, &
      isColumnOrder7_p
  end interface

  private :: &
    isRowOrder1_p, &
    isRowOrder2_p, &
    isRowOrder3_p, &
    isRowOrder4_p, &
    isRowOrder5_p, &
    isRowOrder6_p, &
    isRowOrder7_p

  interface isRowOrder
    module procedure &
      isRowOrder1_p, &
      isRowOrder2_p, &
      isRowOrder3_p, &
      isRowOrder4_p, &
      isRowOrder5_p, &
      isRowOrder6_p, &
      isRowOrder7_p
  end interface

  private :: &
    is_null1_p, &
    is_null2_p, &
    is_null3_p, &
    is_null4_p, &
    is_null5_p, &
    is_null6_p, &
    is_null7_p

  interface is_null
    module procedure &
      is_null1_p, &
      is_null2_p, &
      is_null3_p, &
      is_null4_p, &
      is_null5_p, &
      is_null6_p, &
      is_null7_p
  end interface

  private :: &
    lower1_p, &
    lower2_p, &
    lower3_p, &
    lower4_p, &
    lower5_p, &
    lower6_p, &
    lower7_p

  interface lower
    module procedure &
      lower1_p, &
      lower2_p, &
      lower3_p, &
      lower4_p, &
      lower5_p, &
      lower6_p, &
      lower7_p
  end interface

  private :: &
    not_null1_p, &
    not_null2_p, &
    not_null3_p, &
    not_null4_p, &
    not_null5_p, &
    not_null6_p, &
    not_null7_p

  interface not_null
    module procedure &
      not_null1_p, &
      not_null2_p, &
      not_null3_p, &
      not_null4_p, &
      not_null5_p, &
      not_null6_p, &
      not_null7_p
  end interface

  private :: &
    set_null1_p, &
    set_null2_p, &
    set_null3_p, &
    set_null4_p, &
    set_null5_p, &
    set_null6_p, &
    set_null7_p

  interface set_null
    module procedure &
      set_null1_p, &
      set_null2_p, &
      set_null3_p, &
      set_null4_p, &
      set_null5_p, &
      set_null6_p, &
      set_null7_p
  end interface

  private :: &
    smartCopy1_p, &
    smartCopy2_p, &
    smartCopy3_p, &
    smartCopy4_p, &
    smartCopy5_p, &
    smartCopy6_p, &
    smartCopy7_p

  interface smartCopy
    module procedure &
      smartCopy1_p, &
      smartCopy2_p, &
      smartCopy3_p, &
      smartCopy4_p, &
      smartCopy5_p, &
      smartCopy6_p, &
      smartCopy7_p
  end interface

  private :: &
    stride1_p, &
    stride2_p, &
    stride3_p, &
    stride4_p, &
    stride5_p, &
    stride6_p, &
    stride7_p

  interface stride
    module procedure &
      stride1_p, &
      stride2_p, &
      stride3_p, &
      stride4_p, &
      stride5_p, &
      stride6_p, &
      stride7_p
  end interface

  private :: &
    upper1_p, &
    upper2_p, &
    upper3_p, &
    upper4_p, &
    upper5_p, &
    upper6_p, &
    upper7_p

  interface upper
    module procedure &
      upper1_p, &
      upper2_p, &
      upper3_p, &
      upper4_p, &
      upper5_p, &
      upper6_p, &
      upper7_p
  end interface

  private :: &
    length1_p, &
    length2_p, &
    length3_p, &
    length4_p, &
    length5_p, &
    length6_p, &
    length7_p

  interface length
    module procedure &
      length1_p, &
      length2_p, &
      length3_p, &
      length4_p, &
      length5_p, &
      length6_p, &
      length7_p
  end interface

  private :: create1d1_p

  interface create1d
    module procedure create1d1_p
  end interface

  private :: create2dRow2_p

  interface create2dRow
    module procedure create2dRow2_p
  end interface

  private :: create2dCol2_p

  interface create2dCol
    module procedure create2dCol2_p
  end interface

  private :: &
    slice11_p, &
    slice12_p, &
    slice22_p, &
    slice13_p, &
    slice23_p, &
    slice33_p, &
    slice14_p, &
    slice24_p, &
    slice34_p, &
    slice44_p, &
    slice15_p, &
    slice25_p, &
    slice35_p, &
    slice45_p, &
    slice55_p, &
    slice16_p, &
    slice26_p, &
    slice36_p, &
    slice46_p, &
    slice56_p, &
    slice66_p, &
    slice17_p, &
    slice27_p, &
    slice37_p, &
    slice47_p, &
    slice57_p, &
    slice67_p, &
    slice77_p

  interface slice
    module procedure &
      slice11_p, &
      slice12_p, &
      slice22_p, &
      slice13_p, &
      slice23_p, &
      slice33_p, &
      slice14_p, &
      slice24_p, &
      slice34_p, &
      slice44_p, &
      slice15_p, &
      slice25_p, &
      slice35_p, &
      slice45_p, &
      slice55_p, &
      slice16_p, &
      slice26_p, &
      slice36_p, &
      slice46_p, &
      slice56_p, &
      slice66_p, &
      slice17_p, &
      slice27_p, &
      slice37_p, &
      slice47_p, &
      slice57_p, &
      slice67_p, &
      slice77_p
  end interface

  private :: &
    getg1_p, &
    getg2_p, &
    getg3_p, &
    getg4_p, &
    getg5_p, &
    getg6_p, &
    getg7_p

  private :: &
    get1_p, &
    get2_p, &
    get3_p, &
    get4_p, &
    get5_p, &
    get6_p, &
    get7_p

  interface get
    module procedure &
      getg1_p, &
      get1_p, &
      getg2_p, &
      get2_p, &
      getg3_p, &
      get3_p, &
      getg4_p, &
      get4_p, &
      getg5_p, &
      get5_p, &
      getg6_p, &
      get6_p, &
      getg7_p, &
    get7_p
  end interface

  private :: &
    setg1_p, &
    setg2_p, &
    setg3_p, &
    setg4_p, &
    setg5_p, &
    setg6_p, &
    setg7_p

  private :: &
    set1_p, &
    set2_p, &
    set3_p, &
    set4_p, &
    set5_p, &
    set6_p, &
    set7_p

  interface set
    module procedure &
      setg1_p, &
      set1_p, &
      setg2_p, &
      set2_p, &
      setg3_p, &
      set3_p, &
      setg4_p, &
      set4_p, &
      setg5_p, &
      set5_p, &
      setg6_p, &
      set6_p, &
      setg7_p, &
    set7_p
  end interface


  private :: &
    casthplsupport_BlockCyclicDistArray2dDouble1dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble2dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble3dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble4dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble5dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble6dToGeneric_p, &
    casthplsupport_BlockCyclicDistArray2dDouble7dToGeneric_p

  private :: &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_1d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_2d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_3d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_4d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_5d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_6d_p, &
    cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_7d_p
  interface cast
    module procedure &
      casthplsupport_BlockCyclicDistArray2dDouble1dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble2dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble3dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble4dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble5dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble6dToGeneric_p, &
      casthplsupport_BlockCyclicDistArray2dDouble7dToGeneric_p
    module procedure &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_1d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_2d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_3d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_4d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_5d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_6d_p, &
      cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_7d_p
  end interface


contains


  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(1, lower, upper)
    array%d_array = cptr
  end subroutine createCol1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(1, lower, upper)
    array%d_array = cptr
  end subroutine createRow1_p

  subroutine create1d1_p(len, array)
    integer (kind=sidl_int), intent(in) :: len
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_create1d(len) bind(c,         &
        name="sidl_interface__array_create1d")
        use sidl
        use iso_c_binding
        integer(c_int), value :: len
      end function sidl_interface__array_create1d
    end interface
    cptr = sidl_interface__array_create1d(len)
    array%d_array = cptr
  end subroutine create1d1_p

  subroutine copy1_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy1_p

  subroutine ensure1_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 1, ordering)
    result%d_array = cptr
  end subroutine ensure1_p

  subroutine slice11_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in)  :: src
    integer (kind=sidl_int), dimension(1), intent(in) :: numElem
    integer (kind=sidl_int), dimension(1), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice11_p

  subroutine getg1_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg1_p

  subroutine setg1_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg1_p

  subroutine get1_p(array, &
      i1, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get1(array, i1) bind(c,       &
        name="sidl_interface__array_get1")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
      end function sidl_interface__array_get1
    end interface
    bindc_val = sidl_interface__array_get1(array%d_array, i1)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get1_p

  subroutine set1_p(array, &
      i1, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set1(array, i1, value) bind(c,          &
        name="sidl_interface__array_set1")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set1
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set1(array%d_array, i1, bindc_val)
  end subroutine set1_p

  subroutine smartCopy1_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 1
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy1_p

  logical function  isColumnOrder1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder1_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder1_p = .true.
    end if
  end function isColumnOrder1_p

  logical function  isRowOrder1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder1_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder1_p = .true.
    end if
  end function isRowOrder1_p

  integer (kind=sidl_int) function  dimen1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen1_p = sidl_interface__array_dimen(array%d_array)
  end function dimen1_p

  integer (kind=sidl_int) function  stride1_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride1_p = sidl_interface__array_stride(array%d_array, index)
  end function stride1_p

  integer (kind=sidl_int) function  lower1_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower1_p = sidl_interface__array_lower(array%d_array, index)
  end function lower1_p

  integer (kind=sidl_int) function  upper1_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper1_p = sidl_interface__array_upper(array%d_array, index)
  end function upper1_p

  integer (kind=sidl_int) function  length1_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length1_p = sidl_interface__array_length(array%d_array, index)
  end function length1_p

  subroutine  addRef1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
  end subroutine addRef1_p

  subroutine  deleteRef1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
  end subroutine deleteRef1_p

  logical function is_null1_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    is_null1_p = (.not. c_associated(array%d_array))
  end function is_null1_p

  logical function not_null1_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: array
    not_null1_p = c_associated(array%d_array)
  end function not_null1_p

  subroutine set_null1_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(2, lower, upper)
    array%d_array = cptr
  end subroutine createCol2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(2, lower, upper)
    array%d_array = cptr
  end subroutine createRow2_p

  subroutine create2dCol2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_create2dCol(m, n) bind(c,     &
        name="sidl_interface__array_create2dCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: m
        integer(c_int32_t), value :: n
      end function sidl_interface__array_create2dCol
    end interface
    cptr = sidl_interface__array_create2dCol(m, n)
    array%d_array = cptr
  end subroutine create2dCol2_p

  subroutine create2dRow2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_create2dRow(m, n) bind(c,     &
        name="sidl_interface__array_create2dRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: m
        integer(c_int32_t), value :: n
      end function sidl_interface__array_create2dRow
    end interface
    cptr = sidl_interface__array_create2dRow(m, n)
    array%d_array = cptr
  end subroutine create2dRow2_p

  subroutine copy2_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy2_p

  subroutine ensure2_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 2, ordering)
    result%d_array = cptr
  end subroutine ensure2_p

  subroutine slice12_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice12_p

  subroutine slice22_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice22_p

  subroutine getg2_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg2_p

  subroutine setg2_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg2_p

  subroutine get2_p(array, &
      i1, &
      i2, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get2(array, i1, i2) bind(c,   &
        name="sidl_interface__array_get2")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
      end function sidl_interface__array_get2
    end interface
    bindc_val = sidl_interface__array_get2(array%d_array, i1, i2)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get2_p

  subroutine set2_p(array, &
      i1, &
      i2, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set2(array, i1, i2, value) bind(c,      &
        name="sidl_interface__array_set2")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set2
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set2(array%d_array, i1, i2, bindc_val)
  end subroutine set2_p

  subroutine smartCopy2_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 2
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy2_p

  logical function  isColumnOrder2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder2_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder2_p = .true.
    end if
  end function isColumnOrder2_p

  logical function  isRowOrder2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder2_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder2_p = .true.
    end if
  end function isRowOrder2_p

  integer (kind=sidl_int) function  dimen2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen2_p = sidl_interface__array_dimen(array%d_array)
  end function dimen2_p

  integer (kind=sidl_int) function  stride2_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride2_p = sidl_interface__array_stride(array%d_array, index)
  end function stride2_p

  integer (kind=sidl_int) function  lower2_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower2_p = sidl_interface__array_lower(array%d_array, index)
  end function lower2_p

  integer (kind=sidl_int) function  upper2_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper2_p = sidl_interface__array_upper(array%d_array, index)
  end function upper2_p

  integer (kind=sidl_int) function  length2_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length2_p = sidl_interface__array_length(array%d_array, index)
  end function length2_p

  subroutine  addRef2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
  end subroutine addRef2_p

  subroutine  deleteRef2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
  end subroutine deleteRef2_p

  logical function is_null2_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    is_null2_p = (.not. c_associated(array%d_array))
  end function is_null2_p

  logical function not_null2_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: array
    not_null2_p = c_associated(array%d_array)
  end function not_null2_p

  subroutine set_null2_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(3, lower, upper)
    array%d_array = cptr
  end subroutine createCol3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(3, lower, upper)
    array%d_array = cptr
  end subroutine createRow3_p

  subroutine copy3_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy3_p

  subroutine ensure3_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 3, ordering)
    result%d_array = cptr
  end subroutine ensure3_p

  subroutine slice13_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice13_p

  subroutine slice23_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice23_p

  subroutine slice33_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 3, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice33_p

  subroutine getg3_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg3_p

  subroutine setg3_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg3_p

  subroutine get3_p(array, &
      i1, &
      i2, &
      i3, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get3(array, i1, i2, i3) bind( &
        c, name="sidl_interface__array_get3")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
      end function sidl_interface__array_get3
    end interface
    bindc_val = sidl_interface__array_get3(array%d_array, i1, i2, i3)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get3_p

  subroutine set3_p(array, &
      i1, &
      i2, &
      i3, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set3(array, i1, i2, i3, value) bind(c,  &
        name="sidl_interface__array_set3")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set3
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set3(array%d_array, i1, i2, i3, bindc_val)
  end subroutine set3_p

  subroutine smartCopy3_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 3
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy3_p

  logical function  isColumnOrder3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder3_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder3_p = .true.
    end if
  end function isColumnOrder3_p

  logical function  isRowOrder3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder3_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder3_p = .true.
    end if
  end function isRowOrder3_p

  integer (kind=sidl_int) function  dimen3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen3_p = sidl_interface__array_dimen(array%d_array)
  end function dimen3_p

  integer (kind=sidl_int) function  stride3_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride3_p = sidl_interface__array_stride(array%d_array, index)
  end function stride3_p

  integer (kind=sidl_int) function  lower3_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower3_p = sidl_interface__array_lower(array%d_array, index)
  end function lower3_p

  integer (kind=sidl_int) function  upper3_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper3_p = sidl_interface__array_upper(array%d_array, index)
  end function upper3_p

  integer (kind=sidl_int) function  length3_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length3_p = sidl_interface__array_length(array%d_array, index)
  end function length3_p

  subroutine  addRef3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
  end subroutine addRef3_p

  subroutine  deleteRef3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
  end subroutine deleteRef3_p

  logical function is_null3_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    is_null3_p = (.not. c_associated(array%d_array))
  end function is_null3_p

  logical function not_null3_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: array
    not_null3_p = c_associated(array%d_array)
  end function not_null3_p

  subroutine set_null3_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(4, lower, upper)
    array%d_array = cptr
  end subroutine createCol4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(4, lower, upper)
    array%d_array = cptr
  end subroutine createRow4_p

  subroutine copy4_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy4_p

  subroutine ensure4_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 4, ordering)
    result%d_array = cptr
  end subroutine ensure4_p

  subroutine slice14_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice14_p

  subroutine slice24_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice24_p

  subroutine slice34_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 3, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice34_p

  subroutine slice44_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 4, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice44_p

  subroutine getg4_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg4_p

  subroutine setg4_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg4_p

  subroutine get4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get4(array, i1, i2, i3, i4)   &
        bind(c, name="sidl_interface__array_get4")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
      end function sidl_interface__array_get4
    end interface
    bindc_val = sidl_interface__array_get4(array%d_array, i1, i2, i3, i4)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get4_p

  subroutine set4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set4(array, i1, i2, i3, i4, value)      &
        bind(c, name="sidl_interface__array_set4")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set4
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set4(array%d_array, i1, i2, i3, i4, bindc_val)
  end subroutine set4_p

  subroutine smartCopy4_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 4
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy4_p

  logical function  isColumnOrder4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder4_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder4_p = .true.
    end if
  end function isColumnOrder4_p

  logical function  isRowOrder4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder4_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder4_p = .true.
    end if
  end function isRowOrder4_p

  integer (kind=sidl_int) function  dimen4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen4_p = sidl_interface__array_dimen(array%d_array)
  end function dimen4_p

  integer (kind=sidl_int) function  stride4_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride4_p = sidl_interface__array_stride(array%d_array, index)
  end function stride4_p

  integer (kind=sidl_int) function  lower4_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower4_p = sidl_interface__array_lower(array%d_array, index)
  end function lower4_p

  integer (kind=sidl_int) function  upper4_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper4_p = sidl_interface__array_upper(array%d_array, index)
  end function upper4_p

  integer (kind=sidl_int) function  length4_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length4_p = sidl_interface__array_length(array%d_array, index)
  end function length4_p

  subroutine  addRef4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
  end subroutine addRef4_p

  subroutine  deleteRef4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
  end subroutine deleteRef4_p

  logical function is_null4_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    is_null4_p = (.not. c_associated(array%d_array))
  end function is_null4_p

  logical function not_null4_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: array
    not_null4_p = c_associated(array%d_array)
  end function not_null4_p

  subroutine set_null4_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(5, lower, upper)
    array%d_array = cptr
  end subroutine createCol5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(5, lower, upper)
    array%d_array = cptr
  end subroutine createRow5_p

  subroutine copy5_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy5_p

  subroutine ensure5_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 5, ordering)
    result%d_array = cptr
  end subroutine ensure5_p

  subroutine slice15_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice15_p

  subroutine slice25_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice25_p

  subroutine slice35_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 3, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice35_p

  subroutine slice45_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 4, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice45_p

  subroutine slice55_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 5, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice55_p

  subroutine getg5_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg5_p

  subroutine setg5_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg5_p

  subroutine get5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get5(array, i1, i2, i3, i4,   &
        i5) bind(c, name="sidl_interface__array_get5")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
      end function sidl_interface__array_get5
    end interface
    bindc_val = sidl_interface__array_get5(array%d_array, i1, i2, i3, i4, i5)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get5_p

  subroutine set5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set5(array, i1, i2, i3, i4, i5, value)  &
        bind(c, name="sidl_interface__array_set5")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set5
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set5(array%d_array, i1, i2, i3, i4, i5,         &
      bindc_val)
  end subroutine set5_p

  subroutine smartCopy5_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 5
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy5_p

  logical function  isColumnOrder5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder5_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder5_p = .true.
    end if
  end function isColumnOrder5_p

  logical function  isRowOrder5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder5_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder5_p = .true.
    end if
  end function isRowOrder5_p

  integer (kind=sidl_int) function  dimen5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen5_p = sidl_interface__array_dimen(array%d_array)
  end function dimen5_p

  integer (kind=sidl_int) function  stride5_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride5_p = sidl_interface__array_stride(array%d_array, index)
  end function stride5_p

  integer (kind=sidl_int) function  lower5_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower5_p = sidl_interface__array_lower(array%d_array, index)
  end function lower5_p

  integer (kind=sidl_int) function  upper5_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper5_p = sidl_interface__array_upper(array%d_array, index)
  end function upper5_p

  integer (kind=sidl_int) function  length5_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length5_p = sidl_interface__array_length(array%d_array, index)
  end function length5_p

  subroutine  addRef5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
  end subroutine addRef5_p

  subroutine  deleteRef5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
  end subroutine deleteRef5_p

  logical function is_null5_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    is_null5_p = (.not. c_associated(array%d_array))
  end function is_null5_p

  logical function not_null5_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: array
    not_null5_p = c_associated(array%d_array)
  end function not_null5_p

  subroutine set_null5_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(6, lower, upper)
    array%d_array = cptr
  end subroutine createCol6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(6, lower, upper)
    array%d_array = cptr
  end subroutine createRow6_p

  subroutine copy6_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy6_p

  subroutine ensure6_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 6, ordering)
    result%d_array = cptr
  end subroutine ensure6_p

  subroutine slice16_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice16_p

  subroutine slice26_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice26_p

  subroutine slice36_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 3, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice36_p

  subroutine slice46_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 4, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice46_p

  subroutine slice56_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 5, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice56_p

  subroutine slice66_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 6, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice66_p

  subroutine getg6_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg6_p

  subroutine setg6_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg6_p

  subroutine get6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get6(array, i1, i2, i3, i4,   &
        i5, i6) bind(c, name="sidl_interface__array_get6")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
        integer(c_int32_t), value :: i6
      end function sidl_interface__array_get6
    end interface
    bindc_val = sidl_interface__array_get6(array%d_array, i1, i2, i3, i4, i5,  &
      i6)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get6_p

  subroutine set6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set6(array, i1, i2, i3, i4, i5, i6,     &
        value) bind(c, name="sidl_interface__array_set6")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
        integer(c_int32_t), value :: i6
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set6
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set6(array%d_array, i1, i2, i3, i4, i5, i6,     &
      bindc_val)
  end subroutine set6_p

  subroutine smartCopy6_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 6
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy6_p

  logical function  isColumnOrder6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder6_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder6_p = .true.
    end if
  end function isColumnOrder6_p

  logical function  isRowOrder6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder6_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder6_p = .true.
    end if
  end function isRowOrder6_p

  integer (kind=sidl_int) function  dimen6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen6_p = sidl_interface__array_dimen(array%d_array)
  end function dimen6_p

  integer (kind=sidl_int) function  stride6_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride6_p = sidl_interface__array_stride(array%d_array, index)
  end function stride6_p

  integer (kind=sidl_int) function  lower6_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower6_p = sidl_interface__array_lower(array%d_array, index)
  end function lower6_p

  integer (kind=sidl_int) function  upper6_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper6_p = sidl_interface__array_upper(array%d_array, index)
  end function upper6_p

  integer (kind=sidl_int) function  length6_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length6_p = sidl_interface__array_length(array%d_array, index)
  end function length6_p

  subroutine  addRef6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
  end subroutine addRef6_p

  subroutine  deleteRef6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
  end subroutine deleteRef6_p

  logical function is_null6_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    is_null6_p = (.not. c_associated(array%d_array))
  end function is_null6_p

  logical function not_null6_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: array
    not_null6_p = c_associated(array%d_array)
  end function not_null6_p

  subroutine set_null6_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createCol(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createCol")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createCol
    end interface
    cptr = sidl_interface__array_createCol(7, lower, upper)
    array%d_array = cptr
  end subroutine createCol7_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: array
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_createRow(dimen, lower,       &
        upper) bind(c, name="sidl_interface__array_createRow")
        use sidl
        use iso_c_binding
        integer(c_int32_t), value :: dimen
        integer(c_int32_t), dimension(*) :: lower
        integer(c_int32_t), dimension(*) :: upper
      end function sidl_interface__array_createRow
    end interface
    cptr = sidl_interface__array_createRow(7, lower, upper)
    array%d_array = cptr
  end subroutine createRow7_p

  subroutine copy7_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: dest
    interface
      subroutine sidl_interface__array_copy(src, dst) bind(c,                  &
        name="sidl_interface__array_copy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        type(c_ptr), value :: dst
      end subroutine sidl_interface__array_copy
    end interface
    call sidl_interface__array_copy(src%d_array, dest%d_array)
  end subroutine copy7_p

  subroutine ensure7_p(src, dim, ordering, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim
    integer(kind(sidl_general_order)), intent(in) :: ordering
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_ensure(src, dimen, dst) bind( &
        c, name="sidl_interface__array_ensure")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), value :: dst
      end function sidl_interface__array_ensure
    end interface
    cptr = sidl_interface__array_ensure(src%d_array, 7, ordering)
    result%d_array = cptr
  end subroutine ensure7_p

  subroutine slice17_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 1, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice17_p

  subroutine slice27_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 2, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice27_p

  subroutine slice37_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 3, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice37_p

  subroutine slice47_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 4, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice47_p

  subroutine slice57_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 5, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice57_p

  subroutine slice67_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 6, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice67_p

  subroutine slice77_p(src, numElem, srcStart, srcStride, newLower, result)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_slice(src, dimen, numElem,    &
        srcStart, srcStride, newStart) bind(c,                                 &
        name="sidl_interface__array_slice")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
        integer(c_int32_t), value :: dimen
        integer(c_int), dimension(*) :: numElem
        integer(c_int), dimension(*) :: srcStart
        integer(c_int), dimension(*) :: srcStride
        integer(c_int), dimension(*) :: newStart
      end function sidl_interface__array_slice
    end interface
    cptr = sidl_interface__array_slice(src%d_array, 7, numElem, srcStart,      &
      srcStride, newLower)
    result%d_array = cptr
  end subroutine slice77_p

  subroutine getg7_p(array, index, value)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: value
    type(c_ptr) :: bindc_value
    interface
      type(c_ptr) function sidl_interface__array_get(array, indices) bind(c,   &
        name="sidl_interface__array_get")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
      end function sidl_interface__array_get
    end interface
    bindc_value = sidl_interface__array_get(array%d_array, index)
    value%d_ior = bindc_value
    call cache_epv(value)
  end subroutine getg7_p

  subroutine setg7_p(array, index, val)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(inout) :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set(array, indices, value) bind(c,      &
        name="sidl_interface__array_set")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), dimension(*) :: indices
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set(array%d_array, index, bindc_val)
  end subroutine setg7_p

  subroutine get7_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: val
    type(c_ptr) :: bindc_val
    interface
      type(c_ptr) function sidl_interface__array_get7(array, i1, i2, i3, i4,   &
        i5, i6, i7) bind(c, name="sidl_interface__array_get7")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
        integer(c_int32_t), value :: i6
        integer(c_int32_t), value :: i7
      end function sidl_interface__array_get7
    end interface
    bindc_val = sidl_interface__array_get7(array%d_array, i1, i2, i3, i4, i5,  &
      i6, i7)
    val%d_ior = bindc_val
    call cache_epv(val)
  end subroutine get7_p

  subroutine set7_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      val)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d) :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: val
    type(c_ptr) :: bindc_val
    interface
      subroutine sidl_interface__array_set7(array, i1, i2, i3, i4, i5, i6, i7, &
        value) bind(c, name="sidl_interface__array_set7")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: i1
        integer(c_int32_t), value :: i2
        integer(c_int32_t), value :: i3
        integer(c_int32_t), value :: i4
        integer(c_int32_t), value :: i5
        integer(c_int32_t), value :: i6
        integer(c_int32_t), value :: i7
        type(c_ptr), value :: value
      end subroutine sidl_interface__array_set7
    end interface
    bindc_val = val%d_ior
    call sidl_interface__array_set7(array%d_array, i1, i2, i3, i4, i5, i6, i7, &
      bindc_val)
  end subroutine set7_p

  subroutine smartCopy7_p(src, dest)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: src
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: dest
    integer(sidl_int) :: dim
    type(c_ptr) :: cptr = c_null_ptr
    interface
      type(c_ptr) function sidl_interface__array_smartCopy(src) bind(c,        &
        name="sidl_interface__array_smartCopy")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_smartCopy
    end interface
    dim = 7
    cptr = sidl_interface__array_smartCopy(src%d_array)
    dest%d_array = cptr
  end subroutine smartCopy7_p

  logical function  isColumnOrder7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isColumnOrder(src) bind(c, &
        name="sidl_interface__array_isColumnOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isColumnOrder
    end interface
    isColumnOrder7_p = .false.
    if(sidl_interface__array_isColumnOrder(array%d_array) .ne. 0) then
      isColumnOrder7_p = .true.
    end if
  end function isColumnOrder7_p

  logical function  isRowOrder7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    interface
      integer(c_int) function sidl_interface__array_isRowOrder(src) bind(c,    &
        name="sidl_interface__array_isRowOrder")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_isRowOrder
    end interface
    isRowOrder7_p = .false.
    if(sidl_interface__array_isRowOrder(array%d_array) .ne. 0) then
      isRowOrder7_p = .true.
    end if
  end function isRowOrder7_p

  integer (kind=sidl_int) function  dimen7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    interface
      integer(c_int32_t) function sidl_interface__array_dimen(src) bind(c,     &
        name="sidl_interface__array_dimen")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: src
      end function sidl_interface__array_dimen
    end interface
    dimen7_p = sidl_interface__array_dimen(array%d_array)
  end function dimen7_p

  integer (kind=sidl_int) function  stride7_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_stride(array, index)   &
        bind(c, name="sidl_interface__array_stride")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_stride
    end interface
    stride7_p = sidl_interface__array_stride(array%d_array, index)
  end function stride7_p

  integer (kind=sidl_int) function  lower7_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_lower(array, index)    &
        bind(c, name="sidl_interface__array_lower")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_lower
    end interface
    lower7_p = sidl_interface__array_lower(array%d_array, index)
  end function lower7_p

  integer (kind=sidl_int) function  upper7_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_upper(array, index)    &
        bind(c, name="sidl_interface__array_upper")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_upper
    end interface
    upper7_p = sidl_interface__array_upper(array%d_array, index)
  end function upper7_p

  integer (kind=sidl_int) function  length7_p(array, index)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    integer (kind=sidl_int), intent(in) :: index
    interface
      integer(c_int32_t) function sidl_interface__array_length(array, index)   &
        bind(c, name="sidl_interface__array_length")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
        integer(c_int32_t), value :: index
      end function sidl_interface__array_length
    end interface
    length7_p = sidl_interface__array_length(array%d_array, index)
  end function length7_p

  subroutine  addRef7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
  end subroutine addRef7_p

  subroutine  deleteRef7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
  end subroutine deleteRef7_p

  logical function is_null7_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    is_null7_p = (.not. c_associated(array%d_array))
  end function is_null7_p

  logical function not_null7_p(array)
    use, intrinsic :: iso_c_binding
    use :: sidl
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: array
    not_null7_p = c_associated(array%d_array)
  end function not_null7_p

  subroutine set_null7_p(array)
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(inout) :: array
    array%d_array = c_null_ptr
  end subroutine set_null7_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble1dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble1dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble2dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble2dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble3dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble3dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble4dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble4dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble5dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble5dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble6dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble6dToGeneric_p

  subroutine casthplsupport_BlockCyclicDistArray2dDouble7dToGeneric_p(oldType, &
    newType)
    type(sidl__array), intent(out) :: newType
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine casthplsupport_BlockCyclicDistArray2dDouble7dToGeneric_p


  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_1d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_1d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 1) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_1d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_2d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_2d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 2) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_2d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_3d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_3d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 3) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_3d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_4d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_4d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 4) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_4d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_5d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_5d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 5) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_5d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_6d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_6d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 6) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_6d_p

  subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_7d_p(from,   &
    to)
    type(c_ptr), intent(in) :: from
    type(hplsupport_BlockCyclicDistArray2dDouble_7d), intent(out) :: to
    interface
      type(c_ptr) function sidl_interface__array_cast(array) bind(c,           &
        name="sidl_interface__array_cast")
        use sidl
        use iso_c_binding
        type(c_ptr), value :: array
      end function sidl_interface__array_cast
    end interface
    to%d_array = sidl_interface__array_cast(from)
    if(not_null(to) .and. dimen(to) .ne. 7) call set_null(to)
  end subroutine cast_cptr_to_hplsupport_BlockCyclicDistArray2dDouble_7d_p


end module hplsupport_BlockCyclicDistArray2dDouble_array
