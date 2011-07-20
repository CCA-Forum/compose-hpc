! 
! File:          hpcc_ParallelTranspose_fSkelf.F03
! Symbol:        hpcc.ParallelTranspose-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 0  )
! Description:   Server-side Fortran glue code for hpcc.ParallelTranspose
! 
! WARNING: Automatically generated; changes will be lost
! 
! 


! 
! Symbol "hpcc.ParallelTranspose" (version 0.1)
! 


#include "hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h"
#include "sidl_RuntimeException_fAbbrev.h"
#include "sidl_ClassInfo_fAbbrev.h"
#include "sidl_BaseClass_fAbbrev.h"
#include "sidl_BaseInterface_fAbbrev.h"
#include "hpcc_ParallelTranspose_fAbbrev.h"
#include "sidl_BaseException_fAbbrev.h"
#include "sidl_rmi_Call_fAbbrev.h"
#include "sidl_rmi_Return_fAbbrev.h"
#include "sidl_rmi_Ticket_fAbbrev.h"

subroutine hpcc_ParallelTranspose_wrap(obj, self, exception)
  use hpcc_ParallelTranspose_Impl
  use, intrinsic :: iso_c_binding
  implicit none
  type(hpcc_ParallelTranspose_impl_t), target :: obj
  type(hpcc_ParallelTranspose_t) :: self
  type(sidl_BaseInterface_t) :: exception
  interface
    subroutine set_data_ptr(ior, ptr) bind(c,                                  &
      name="hpcc_ParallelTranspose_setData")
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
end subroutine hpcc_ParallelTranspose_wrap


module hpcc_ParallelTranspose_fSkelf
  use, intrinsic :: iso_c_binding
  use sidl_BaseInterface_type
  use sidl_RuntimeException_type
  use hpcc_ParallelTranspose_type
  use hplsupport_BlockCyclicDistArray2dDouble_type
  use hpcc_ParallelTranspose_Impl
  implicit none
contains
  subroutine hpcc_ParallelTranspose_impl_cast(c_ior, impl)
    implicit none
    type(c_ptr), intent(in) :: c_ior
    type(hpcc_ParallelTranspose_impl_t), pointer, intent(out) :: impl
    type(c_ptr) :: cptr
    interface
      type(c_ptr) function get_data_ptr(ior) bind(c,                           &
        name="hpcc_ParallelTranspose_getData")
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
  end subroutine hpcc_ParallelTranspose_impl_cast

  subroutine hpcc_ParallelTranspose__ctor_fskel_c(self, exception) bind(c)
    !  in hpcc.ParallelTranspose self
    type(c_ptr), value :: self
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hpcc_ParallelTranspose_impl_t), pointer :: proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception
    interface
      subroutine set_data_ptr(ior, ptr) bind(c,                                &
        name="hpcc_ParallelTranspose_setData")
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

    call hpcc_ParallelTranspose_impl_cast(self, proxy_self)

    call ctor_impl(proxy_self, proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hpcc_ParallelTranspose__ctor_fskel_c


  subroutine hpcc_ParallelTranspose__dtor_fskel_c(self, exception) bind(c)
    !  in hpcc.ParallelTranspose self
    type(c_ptr), value :: self
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hpcc_ParallelTranspose_impl_t), pointer :: proxy_self
    type(sidl_BaseInterface_t) :: proxy_exception

    call hpcc_ParallelTranspose_impl_cast(self, proxy_self)

    call dtor_impl(proxy_self, proxy_exception)

    exception = proxy_exception%d_ior

    ! 
    ! deallocation for user-defined data type
    ! 

    deallocate(proxy_self)
  end subroutine hpcc_ParallelTranspose__dtor_fskel_c


  subroutine hpcc_ParallelTranspose__load_fskel_c(exception) bind(c,           &
    name="hpcc_ParallelTranspose__load_skel_c")
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(sidl_BaseInterface_t) :: proxy_exception


    call load_impl(proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hpcc_ParallelTranspose__load_fskel_c


  subroutine hpcc_ParallelTranspose_ptransCompute_fskel_c(a, c, beta, i, j,    &
    exception) bind(c)
    !  in hplsupport.BlockCyclicDistArray2dDouble a
    type(c_ptr), value :: a
    !  in hplsupport.BlockCyclicDistArray2dDouble c
    type(c_ptr), value :: c
    !  in double beta
    real(c_double), value :: beta
    !  in int i
    integer(c_int32_t), value :: i
    !  in int j
    integer(c_int32_t), value :: j
    !  out sidl.BaseInterface exception
    type(c_ptr) :: exception
    type(hplsupport_BlockCyclicDistArray2dDouble_t) :: proxy_a
    type(hplsupport_BlockCyclicDistArray2dDouble_t) :: proxy_c
    type(sidl_BaseInterface_t) :: proxy_exception

    proxy_a%d_ior = a
    call cache_epv(proxy_a)
    proxy_c%d_ior = c
    call cache_epv(proxy_c)

    call ptransCompute_impl(proxy_a, proxy_c, beta, i, j, proxy_exception)

    exception = proxy_exception%d_ior
  end subroutine hpcc_ParallelTranspose_ptransCompute_fskel_c


  subroutine hpcc_ParallelTranspose__set_epv_bindc(c_epv) bind(c,              &
    name="hpcc_ParallelTranspose__set_epv_bindc")
    type(c_ptr), value :: c_epv
    type(hpcc_ParallelTranspose_epv_t), pointer  :: f_epv => null()

    call c_f_pointer(c_epv, f_epv)

    f_epv%f__ctor = c_funloc(hpcc_ParallelTranspose__ctor_fskel_c)
    f_epv%f__dtor = c_funloc(hpcc_ParallelTranspose__dtor_fskel_c)
  end subroutine hpcc_ParallelTranspose__set_epv_bindc

  subroutine hpcc_ParallelTranspose__set_sepv_bindc(c_epv) bind(c,             &
    name="hpcc_ParallelTranspose__set_sepv_bindc")
    type(c_ptr), value :: c_epv
    type(hpcc_ParallelTranspose_sepv_t), pointer  :: f_epv => null()

    call c_f_pointer(c_epv, f_epv)

    f_epv%f_ptransCompute = c_funloc(                                          &
      hpcc_ParallelTranspose_ptransCompute_fskel_c)
  end subroutine hpcc_ParallelTranspose__set_sepv_bindc


end module hpcc_ParallelTranspose_fSkelf
