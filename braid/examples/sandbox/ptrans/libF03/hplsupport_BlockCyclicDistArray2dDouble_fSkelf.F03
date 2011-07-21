! 
! File:          hplsupport_BlockCyclicDistArray2dDouble_fSkelf.F03
! Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 7138  trunk)
! Description:   Server-side Fortran glue code for hplsupport.BlockCyclicDistArray2dDouble
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

subroutine hplsupport_BlockCyclicDistArray2dDouble_wrap(obj, self, exception)
  use hplsupport_BlockCyclicDistArray2dDouble_Impl
  use, intrinsic :: iso_c_binding
  implicit none
  type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), target :: obj
  type(hplsupport_BlockCyclicDistArray2dDouble_t) :: self
  type(sidl_BaseInterface_t) :: exception
  interface
    subroutine set_data_ptr(ior, ptr) bind(c,                                  &
      name="hplsupport_BlockCyclicDistArray2dDouble_setData")
      use iso_c_binding
      type(c_ptr), value :: ior
      type(c_ptr), value :: ptr
    end subroutine set_data_ptr
  end interface

  ! 
  ! wrapping of user-defined data type
  ! 

  call new(self, exception)
  call set_data_ptr(self%d_ior, c_loc(obj))
  obj%d_ior = self%d_ior
  call cache_epv(obj)
  call ctor_impl(obj, exception)
end subroutine hplsupport_BlockCyclicDistArray2dDouble_wrap


module hplsupport_BlockCyclicDistArray2dDouble_fSkelf
  use, intrinsic :: iso_c_binding
  use sidl_BaseInterface_type
  use sidl_RuntimeException_type
  use hplsupport_BlockCyclicDistArray2dDouble_type
  use hplsupport_BlockCyclicDistArray2dDouble_Impl
  implicit none
contains
  subroutine hplsupport_BlockCyclicDistArray2dDouble_impl_cast(c_ior, impl)
    implicit none
    type(c_ptr), intent(in) :: c_ior
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer, intent(     &
      out) :: impl
    type(c_ptr) :: cptr
    interface
      type(c_ptr) function get_data_ptr(ior) bind(c,                           &
        name="hplsupport_BlockCyclicDistArray2dDouble_getData")
        use iso_c_binding
        type(c_ptr), value :: ior
      end function get_data_ptr
    end interface
    if(c_associated(c_ior)) then
      cptr = get_data_ptr(c_ior)
      if(c_associated(cptr)) then
        call c_f_pointer(cptr, impl)
      endif
    endif
  end subroutine hplsupport_BlockCyclicDistArray2dDouble_impl_cast

  subroutine hplsupport_BlockCyclicDistArray2dDouble__ctor_fskel_c(self,       &
    exception) bind(c)
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    type(c_ptr), value :: self
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer ::           &
      proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception
    interface
      subroutine set_data_ptr(ior, ptr) bind(c,                                &
        name="hplsupport_BlockCyclicDistArray2dDouble_setData")
        use iso_c_binding
        type(c_ptr), value :: ior
        type(c_ptr), value :: ptr
      end subroutine set_data_ptr
    end interface

    ! 
    ! allocation for user-defined data type
    ! 

    allocate(proxy_self)
    proxy_self%d_ior = self
    call cache_epv(proxy_self)
    call set_data_ptr(self, c_loc(proxy_self))

    call hplsupport_BlockCyclicDistArray2dDouble_impl_cast(self, proxy_self)

    call ctor_impl(proxy_self, proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hplsupport_BlockCyclicDistArray2dDouble__ctor_fskel_c


  subroutine hplsupport_BlockCyclicDistArray2dDouble__dtor_fskel_c(self,       &
    exception) bind(c)
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    type(c_ptr), value :: self
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer ::           &
      proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception

    call hplsupport_BlockCyclicDistArray2dDouble_impl_cast(self, proxy_self)

    call dtor_impl(proxy_self, proxy_exception)

    exception = proxy_exception%d_ior

    ! 
    ! deallocation for user-defined data type
    ! 

    deallocate(proxy_self)
  end subroutine hplsupport_BlockCyclicDistArray2dDouble__dtor_fskel_c


  subroutine hplsupport_BlockCyclicDistArray2dDouble__load_fskel_c(exception)  &
    bind(c, name="hplsupport_BlockCyclicDistArray2dDouble__load_skel_c")
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(sidl_BaseInterface_t) :: proxy_exception


    call load_impl(proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hplsupport_BlockCyclicDistArray2dDouble__load_fskel_c


  subroutine hplsupport_BlockCyclicDistArray2dDouble_initData_fskel_c(self,    &
    data, exception) bind(c)
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    type(c_ptr), value :: self
    !  in opaque data
    type(c_ptr), value :: data
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer ::           &
      proxy_self
    type(sidl_opaque_t) :: proxy_data
    type(sidl_BaseInterface_t) :: proxy_exception

    call hplsupport_BlockCyclicDistArray2dDouble_impl_cast(self, proxy_self)
    proxy_data%value = data

    call initData_impl(proxy_self, proxy_data, proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hplsupport_BlockCyclicDistArray2dDouble_initData_fskel_c


  function hplsupport_BlockCyclicDistArray2dDouble_get_fskel_c(self, idx1,     &
    idx2, exception) result(retval) bind(c)
    !  in hplsupport.BlockCyclicDistArray2dDouble self
    type(c_ptr), value :: self
    !  in int idx1
    integer(c_int32_t), value :: idx1
    !  in int idx2
    integer(c_int32_t), value :: idx2
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    !  function result
    real(c_double) :: retval
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer ::           &
      proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception

    call hplsupport_BlockCyclicDistArray2dDouble_impl_cast(self, proxy_self)

    retval = get_impl(proxy_self, idx1, idx2, proxy_exception)

    exception = proxy_exception%d_ior
  end function hplsupport_BlockCyclicDistArray2dDouble_get_fskel_c


  subroutine hplsupport_BlockCyclicDistArray2dDouble_set_fskel_c(self, newVal, &
    idx1, idx2, exception) bind(c)
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
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), pointer ::           &
      proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception

    call hplsupport_BlockCyclicDistArray2dDouble_impl_cast(self, proxy_self)

    call set_impl(proxy_self, newVal, idx1, idx2, proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hplsupport_BlockCyclicDistArray2dDouble_set_fskel_c


  subroutine hplsupport_BlockCyclicDistArray2dDouble__set_epv_bindc(c_epv)     &
    bind(c, name="hplsupport_BlockCyclicDistArray2dDouble__set_epv_bindc")
    type(c_ptr), value :: c_epv
    type(hplsupport_BlockCyclicDistArray2dDouble_epv_t), pointer  :: f_epv =>  &
      null()

    call c_f_pointer(c_epv, f_epv)

    f_epv%f__ctor = c_funloc(                                                  &
      hplsupport_BlockCyclicDistArray2dDouble__ctor_fskel_c)
    f_epv%f__dtor = c_funloc(                                                  &
      hplsupport_BlockCyclicDistArray2dDouble__dtor_fskel_c)
    f_epv%f_initData = c_funloc(                                               &
      hplsupport_BlockCyclicDistArray2dDouble_initData_fskel_c)
    f_epv%f_get = c_funloc(                                                    &
      hplsupport_BlockCyclicDistArray2dDouble_get_fskel_c)
    f_epv%f_set = c_funloc(                                                    &
      hplsupport_BlockCyclicDistArray2dDouble_set_fskel_c)
  end subroutine hplsupport_BlockCyclicDistArray2dDouble__set_epv_bindc


end module hplsupport_BlockCyclicDistArray2dDouble_fSkelf
