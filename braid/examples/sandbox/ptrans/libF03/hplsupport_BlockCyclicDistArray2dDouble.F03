! 
! File:          hplsupport_BlockCyclicDistArray2dDouble.F03
! Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 7138  trunk)
! Description:   Client-side module for hplsupport.BlockCyclicDistArray2dDouble
! 
! WARNING: Automatically generated; changes will be lost
! 
! 


! 
! Symbol "hplsupport.BlockCyclicDistArray2dDouble" (version 0.1)
! 


#include "sidl_BaseException_fAbbrev.h"
#include "sidl_BaseInterface_fAbbrev.h"
#include "sidl_BaseClass_fAbbrev.h"
#include "sidl_ClassInfo_fAbbrev.h"
#include "sidl_RuntimeException_fAbbrev.h"
#include "hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h"
#include "sidl_rmi_Call_fAbbrev.h"
#include "sidl_rmi_Return_fAbbrev.h"
#include "sidl_rmi_Ticket_fAbbrev.h"

module hplsupport_BlockCyclicDistArray2dDouble

  use sidl
  use sidl_BaseException_type
  use sidl_BaseInterface_type
  use sidl_BaseClass_type
  use sidl_ClassInfo_type
  use sidl_RuntimeException_type
  use hplsupport_BlockCyclicDistArray2dDouble_type
  use sidl_rmi_Call_type
  use sidl_rmi_Return_type
  use sidl_rmi_Ticket_type

  private :: cast_0, cast_1, cast_2, cast_3
  interface cast
    module procedure cast_0, cast_1, cast_2, cast_3
  end interface

  private :: initData_s
  interface initData
    module procedure initData_s
  end interface


  private :: get_s
  interface get
    module procedure get_s
  end interface


  private :: set_s
  interface set
    module procedure set_s
  end interface


  private :: newLocal_s, newRemote_s
  interface new
    module procedure newLocal_s, newRemote_s
  end interface


  private :: rConnect_s
  interface rConnect
    module procedure rConnect_s
  end interface

  private :: addRef_s
  interface addRef
    module procedure addRef_s
  end interface


  private :: deleteRef_s
  interface deleteRef
    module procedure deleteRef_s
  end interface


  private :: isSame_s
  interface isSame
    module procedure isSame_s
  end interface


  private :: isType_s
  interface isType
    module procedure isType_s
  end interface


  private :: getClassInfo_s
  interface getClassInfo
    module procedure getClassInfo_s
  end interface



  private :: exec_s
  interface exec
    module procedure exec_s
  end interface


  private :: getURL_s
  interface getURL
    module procedure getURL_s
  end interface


  private :: isRemote_s
  interface isRemote
    module procedure isRemote_s
  end interface


  private :: isLocal_s
  interface isLocal
    module procedure isLocal_s
  end interface


  private :: set_hooks_s
  interface set_hooks
    module procedure set_hooks_s
  end interface


  private :: set_contracts_s
  interface set_contracts
    module procedure set_contracts_s
  end interface


  private :: dump_stats_s
  interface dump_stats
    module procedure dump_stats_s
  end interface

  private :: not_null_s
  interface not_null
    module procedure not_null_s
  end interface

  private :: is_null_s
  interface is_null
    module procedure is_null_s
  end interface

  private :: set_null_s
  interface set_null
    module procedure set_null_s
  end interface


contains



  recursive subroutine initData_s(self, data, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in opaque data
    type(sidl_opaque_t) , intent(in) :: data
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_initData_c(self,      &
        data, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in opaque data
        type(c_ptr), value :: data
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_initData_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_data
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_initData_c), pointer ::  &
      f_initData

    bindc_self = self%d_ior
    bindc_data = data%value
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_initData, f_initData)
    call f_initData(bindc_self, bindc_data, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine initData_s


  function get_s(self, idx1, idx2, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in int idx1
    integer (kind=sidl_int) , intent(in) :: idx1
    !  in int idx2
    integer (kind=sidl_int) , intent(in) :: idx2
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    real (kind=sidl_double) :: retval

    interface
      real(c_double) function hplsupport_BlockCyclicDistArray2dDouble_get_c(   &
        self, idx1, idx2, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in int idx1
        integer(c_int32_t), value :: idx1
        !  in int idx2
        integer(c_int32_t), value :: idx2
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble_get_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_get_c), pointer :: f_get

    bindc_self = self%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_get, f_get)
    retval = f_get(bindc_self, idx1, idx2, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function get_s


  recursive subroutine set_s(self, newVal, idx1, idx2, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in double newVal
    real (kind=sidl_double) , intent(in) :: newVal
    !  in int idx1
    integer (kind=sidl_int) , intent(in) :: idx1
    !  in int idx2
    integer (kind=sidl_int) , intent(in) :: idx2
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_set_c(self, newVal,   &
        idx1, idx2, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in double newVal
        real(c_double), value :: newVal
        !  in int idx1
        integer(c_int32_t), value :: idx1
        !  in int idx2
        integer(c_int32_t), value :: idx2
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_set_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_set_c), pointer :: f_set

    bindc_self = self%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_set, f_set)
    call f_set(bindc_self, newVal, idx1, idx2, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine set_s


  recursive subroutine newLocal_s(self, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  out hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(out) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_newLocal_c(self,      &
        exception) bind(c,                                                     &
        name="hplsupport_BlockCyclicDistArray2dDouble_newLocal_c")
        use iso_c_binding
        !  out hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr) :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_newLocal_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception

    bindc_self = c_null_ptr
    bindc_exception = c_null_ptr
    call hplsupport_BlockCyclicDistArray2dDouble_newLocal_c(bindc_self,        &
      bindc_exception)
    if(c_associated(bindc_exception)) then
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    else
      self%d_ior = bindc_self
      call cache_epv(self)
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    endif

  end subroutine newLocal_s


  recursive subroutine newRemote_s(self, url, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  out hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_newRemote_c(self,     &
        url, exception) bind(c,                                                &
        name="hplsupport_BlockCyclicDistArray2dDouble_newRemote_c")
        use iso_c_binding
        !  out hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr) :: self
        !  in string url
        character(kind=c_char), dimension(*) :: url
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_newRemote_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception

    bindc_self = c_null_ptr
    bindc_exception = c_null_ptr
    call hplsupport_BlockCyclicDistArray2dDouble_newRemote_c(bindc_self, trim( &
      url) // c_null_char, bindc_exception)
    if(c_associated(bindc_exception)) then
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    else
      self%d_ior = bindc_self
      call cache_epv(self)
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    endif

  end subroutine newRemote_s


  recursive subroutine rConnect_s(self, url, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  out hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_rConnect_c(self, url, &
        exception) bind(c,                                                     &
        name="hplsupport_BlockCyclicDistArray2dDouble_rConnect_c")
        use iso_c_binding
        !  out hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr) :: self
        !  in string url
        character(kind=c_char), dimension(*) :: url
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_rConnect_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception

    bindc_self = c_null_ptr
    bindc_exception = c_null_ptr
    call hplsupport_BlockCyclicDistArray2dDouble_rConnect_c(bindc_self, trim(  &
      url) // c_null_char, bindc_exception)
    if(c_associated(bindc_exception)) then
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    else
      self%d_ior = bindc_self
      call cache_epv(self)
      exception%d_ior = bindc_exception
      call cache_epv(exception)
    endif

  end subroutine rConnect_s


  recursive subroutine addRef_s(self, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_addRef_c(self,        &
        exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_addRef_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_addRef_c), pointer ::    &
      f_addRef

    bindc_self = self%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_addRef, f_addRef)
    call f_addRef(bindc_self, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine addRef_s


  recursive subroutine deleteRef_s(self, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble_deleteRef_c(self,     &
        exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble_deleteRef_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_deleteRef_c), pointer    &
      :: f_deleteRef

    bindc_self = self%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_deleteRef, f_deleteRef)
    call f_deleteRef(bindc_self, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine deleteRef_s


  function isSame_s(self, iobj, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in sidl.BaseInterface iobj
    type(sidl_BaseInterface_t) , intent(in) :: iobj
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    logical :: retval

    interface
      integer(c_int) function                                                  &
        hplsupport_BlockCyclicDistArray2dDouble_isSame_c(self, iobj,           &
        exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in sidl.BaseInterface iobj
        type(c_ptr), value :: iobj
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble_isSame_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_iobj
    integer(c_int) :: bindc_retval
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_isSame_c), pointer ::    &
      f_isSame

    bindc_self = self%d_ior
    bindc_iobj = iobj%d_ior
    bindc_retval = 0
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_isSame, f_isSame)
    bindc_retval = f_isSame(bindc_self, bindc_iobj, bindc_exception)
    retval = .false.
    if(bindc_retval .ne. 0) retval = .true.
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function isSame_s


  function isType_s(self, name, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in string name
    character (len=*) , intent(in) :: name
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    logical :: retval

    interface
      integer(c_int) function                                                  &
        hplsupport_BlockCyclicDistArray2dDouble_isType_c(self, name,           &
        exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in string name
        character(kind=c_char), dimension(*) :: name
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble_isType_c
    end interface
    type(c_ptr) :: bindc_self
    integer(c_int) :: bindc_retval
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_isType_c), pointer ::    &
      f_isType

    bindc_self = self%d_ior
    bindc_retval = 0
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_isType, f_isType)
    bindc_retval = f_isType(bindc_self, trim(name) // c_null_char,             &
      bindc_exception)
    retval = .false.
    if(bindc_retval .ne. 0) retval = .true.
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function isType_s


  function getClassInfo_s(self, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    type(sidl_ClassInfo_t) :: retval

    interface
      type(c_ptr) function                                                     &
        hplsupport_BlockCyclicDistArray2dDouble_getClassInfo_c(self,           &
        exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble_getClassInfo_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_retval
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble_getClassInfo_c),         &
      pointer :: f_getClassInfo

    bindc_self = self%d_ior
    bindc_retval = c_null_ptr
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f_getClassInfo, f_getClassInfo)
    bindc_retval = f_getClassInfo(bindc_self, bindc_exception)
    retval%d_ior = bindc_retval
    call cache_epv(retval)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function getClassInfo_s

  ! 
  ! Static function to cast from hplsupport.BlockCyclicDistArray2dDouble
  ! to sidl.BaseClass.
  ! 

  subroutine cast_0(oldType, newType, exception)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: oldType
    type(sidl_BaseClass_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    interface
      subroutine sidl_BaseClass__cast_c(ref, retval, exception) bind(c,        &
        name="sidl_BaseClass__cast_c")
        use iso_c_binding
        !  inout opaque ref
        type(c_ptr) :: ref
        !  out sidl.BaseClass retval
        type(c_ptr) :: retval
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine sidl_BaseClass__cast_c
    end interface
    call sidl_BaseClass__cast_c(oldType%d_ior, newType%d_ior, exception%d_ior)
    call cache_epv(newType)
    call cache_epv(exception)
  end subroutine cast_0

  ! 
  ! Static function to cast from sidl.BaseClass
  ! to hplsupport.BlockCyclicDistArray2dDouble.
  ! 

  subroutine cast_1(oldType, newType, exception)
    implicit none
    type(sidl_BaseClass_t), intent(in) :: oldType
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__cast_c(ref, retval,  &
        exception) bind(c,                                                     &
        name="hplsupport_BlockCyclicDistArray2dDouble__cast_c")
        use iso_c_binding
        !  inout opaque ref
        type(c_ptr) :: ref
        !  out hplsupport.BlockCyclicDistArray2dDouble retval
        type(c_ptr) :: retval
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__cast_c
    end interface
    call hplsupport_BlockCyclicDistArray2dDouble__cast_c(oldType%d_ior,        &
      newType%d_ior, exception%d_ior)
    call cache_epv(newType)
    call cache_epv(exception)
  end subroutine cast_1

  ! 
  ! Static function to cast from hplsupport.BlockCyclicDistArray2dDouble
  ! to sidl.BaseInterface.
  ! 

  subroutine cast_2(oldType, newType, exception)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: oldType
    type(sidl_BaseInterface_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    interface
      subroutine sidl_BaseInterface__cast_c(ref, retval, exception) bind(c,    &
        name="sidl_BaseInterface__cast_c")
        use iso_c_binding
        !  inout opaque ref
        type(c_ptr) :: ref
        !  out sidl.BaseInterface retval
        type(c_ptr) :: retval
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine sidl_BaseInterface__cast_c
    end interface
    call sidl_BaseInterface__cast_c(oldType%d_ior, newType%d_ior,              &
      exception%d_ior)
    call cache_epv(newType)
    call cache_epv(exception)
  end subroutine cast_2

  ! 
  ! Static function to cast from sidl.BaseInterface
  ! to hplsupport.BlockCyclicDistArray2dDouble.
  ! 

  subroutine cast_3(oldType, newType, exception)
    implicit none
    type(sidl_BaseInterface_t), intent(in) :: oldType
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__cast_c(ref, retval,  &
        exception) bind(c,                                                     &
        name="hplsupport_BlockCyclicDistArray2dDouble__cast_c")
        use iso_c_binding
        !  inout opaque ref
        type(c_ptr) :: ref
        !  out hplsupport.BlockCyclicDistArray2dDouble retval
        type(c_ptr) :: retval
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__cast_c
    end interface
    call hplsupport_BlockCyclicDistArray2dDouble__cast_c(oldType%d_ior,        &
      newType%d_ior, exception%d_ior)
    call cache_epv(newType)
    call cache_epv(exception)
  end subroutine cast_3


  recursive subroutine exec_s(self, methodName, inArgs, outArgs, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in string methodName
    character (len=*) , intent(in) :: methodName
    !  in sidl.rmi.Call inArgs
    type(sidl_rmi_Call_t) , intent(in) :: inArgs
    !  in sidl.rmi.Return outArgs
    type(sidl_rmi_Return_t) , intent(in) :: outArgs
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__exec_c(self,         &
        methodName, inArgs, outArgs, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in string methodName
        character(kind=c_char), dimension(*) :: methodName
        !  in sidl.rmi.Call inArgs
        type(c_ptr), value :: inArgs
        !  in sidl.rmi.Return outArgs
        type(c_ptr), value :: outArgs
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__exec_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_inArgs
    type(c_ptr) :: bindc_outArgs
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__exec_c), pointer ::     &
      f__exec

    bindc_self = self%d_ior
    bindc_inArgs = inArgs%d_ior
    bindc_outArgs = outArgs%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__exec, f__exec)
    call f__exec(bindc_self, trim(methodName) // c_null_char, bindc_inArgs,    &
      bindc_outArgs, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine exec_s

  function getURL_s(self, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    character (len=sidl_f03_str_minsize) :: retval

    interface
      type(c_ptr) function hplsupport_BlockCyclicDistArray2dDouble__getURL_c(  &
        self, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble__getURL_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_retval
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__getURL_c), pointer ::   &
      f__getURL

    bindc_self = self%d_ior
    bindc_retval = c_null_ptr
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__getURL, f__getURL)
    bindc_retval = f__getURL(bindc_self, bindc_exception)
    if (c_associated(bindc_retval)) then
      call sidl_copy_c_str(retval, len(retval, c_size_t), bindc_retval)
      call sidl_String_free(bindc_retval)
    else
      retval = ''
    endif
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function getURL_s

  function isRemote_s(self, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    logical :: retval

    interface
      integer(c_int) function                                                  &
        hplsupport_BlockCyclicDistArray2dDouble__isRemote_c(self, exception)   &
        bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end function hplsupport_BlockCyclicDistArray2dDouble__isRemote_c
    end interface
    type(c_ptr) :: bindc_self
    integer(c_int) :: bindc_retval
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__isRemote_c), pointer    &
      :: f__isRemote

    bindc_self = self%d_ior
    bindc_retval = 0
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__isRemote, f__isRemote)
    bindc_retval = f__isRemote(bindc_self, bindc_exception)
    retval = .false.
    if(bindc_retval .ne. 0) retval = .true.
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function isRemote_s

  function isLocal_s(self, exception) result(retval)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  function return value
    logical :: retval

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__isLocal_c(self,      &
        retval, exception) bind(c,                                             &
        name="hplsupport_BlockCyclicDistArray2dDouble__isLocal_c")
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  out bool retval
        integer(c_int) :: retval
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__isLocal_c
    end interface
    type(c_ptr) :: bindc_self
    integer(c_int) :: bindc_retval
    type(c_ptr) :: bindc_exception

    bindc_self = self%d_ior
    bindc_retval = 0
    bindc_exception = c_null_ptr
    call hplsupport_BlockCyclicDistArray2dDouble__isLocal_c(bindc_self,        &
      bindc_retval, bindc_exception)
    retval = .false.
    if(bindc_retval .ne. 0) retval = .true.
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end function isLocal_s

  recursive subroutine set_hooks_s(self, enable, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in bool enable
    logical , intent(in) :: enable
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__set_hooks_c(self,    &
        enable, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in bool enable
        integer(c_int), value :: enable
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__set_hooks_c
    end interface
    type(c_ptr) :: bindc_self
    integer(c_int) :: bindc_enable
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__set_hooks_c), pointer   &
      :: f__set_hooks

    bindc_self = self%d_ior
    bindc_enable = 0
    if(enable) bindc_enable = 1
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__set_hooks, f__set_hooks)
    call f__set_hooks(bindc_self, bindc_enable, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine set_hooks_s

  recursive subroutine set_contracts_s(self, enable, enfFilename,              &
    resetCounters, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in bool enable
    logical , intent(in) :: enable
    !  in string enfFilename
    character (len=*) , intent(in) :: enfFilename
    !  in bool resetCounters
    logical , intent(in) :: resetCounters
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__set_contracts_c(     &
        self, enable, enfFilename, resetCounters, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in bool enable
        integer(c_int), value :: enable
        !  in string enfFilename
        character(kind=c_char), dimension(*) :: enfFilename
        !  in bool resetCounters
        integer(c_int), value :: resetCounters
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__set_contracts_c
    end interface
    type(c_ptr) :: bindc_self
    integer(c_int) :: bindc_enable
    integer(c_int) :: bindc_resetCounters
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__set_contracts_c),       &
      pointer :: f__set_contracts

    bindc_self = self%d_ior
    bindc_enable = 0
    if(enable) bindc_enable = 1
    bindc_resetCounters = 0
    if(resetCounters) bindc_resetCounters = 1
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__set_contracts, f__set_contracts)
    call f__set_contracts(bindc_self, bindc_enable, trim(enfFilename) //       &
      c_null_char, bindc_resetCounters, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine set_contracts_s

  recursive subroutine dump_stats_s(self, filename, prefix, exception)
    use, intrinsic :: iso_c_binding
    implicit none
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    class(hplsupport_BlockCyclicDistArray2dDouble_t) , intent(in) :: self
    !  in string filename
    character (len=*) , intent(in) :: filename
    !  in string prefix
    character (len=*) , intent(in) :: prefix
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    interface
      subroutine hplsupport_BlockCyclicDistArray2dDouble__dump_stats_c(self,   &
        filename, prefix, exception) bind(c)
        use iso_c_binding
        !  in hplsupport.BlockCyclicDistArray2dDouble self
        type(c_ptr), value :: self
        !  in string filename
        character(kind=c_char), dimension(*) :: filename
        !  in string prefix
        character(kind=c_char), dimension(*) :: prefix
        !  out sidl.BaseInterface exception
        type(c_ptr) :: exception
      end subroutine hplsupport_BlockCyclicDistArray2dDouble__dump_stats_c
    end interface
    type(c_ptr) :: bindc_self
    type(c_ptr) :: bindc_exception
    procedure(hplsupport_BlockCyclicDistArray2dDouble__dump_stats_c), pointer  &
      :: f__dump_stats

    bindc_self = self%d_ior
    bindc_exception = c_null_ptr
    call c_f_procpointer(self%d_epv%f__dump_stats, f__dump_stats)
    call f__dump_stats(bindc_self, trim(filename) // c_null_char, trim(        &
      prefix) // c_null_char, bindc_exception)
    exception%d_ior = bindc_exception
    call cache_epv(exception)

  end subroutine dump_stats_s
  logical function is_null_s(ext)
    class(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: ext
    is_null_s = .not. c_associated(ext%d_ior)
  end function is_null_s

  logical function not_null_s(ext)
    class(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: ext
    not_null_s = c_associated(ext%d_ior)
  end function not_null_s

  subroutine set_null_s(ext)
    class(hplsupport_BlockCyclicDistArray2dDouble_t), intent(out) :: ext
    ext%d_ior = c_null_ptr
    ext%d_epv => null()
  end subroutine set_null_s


end module hplsupport_BlockCyclicDistArray2dDouble
