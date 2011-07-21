! 
! File:          hplsupport_BlockCyclicDistArray2dDouble_Impl.F03
! Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 7138  trunk)
! Description:   Server-side implementation for hplsupport.BlockCyclicDistArray2dDouble
! 
! WARNING: Automatically generated; only changes within splicers preserved
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
! DO-NOT-DELETE splicer.begin(module.include)
! 
! Add additional include statements here
! 

! DO-NOT-DELETE splicer.end(module.include)

module hplsupport_BlockCyclicDistArray2dDouble_Impl
  use sidl
  use sidl_BaseInterface
  use sidl_RuntimeException
  use hplsupport_BlockCyclicDistArray2dDouble
  ! DO-NOT-DELETE splicer.begin(module.use)
  ! 
  ! Add use statements here
  ! 

  ! DO-NOT-DELETE splicer.end(module.use)


  type, extends(hplsupport_BlockCyclicDistArray2dDouble_t) ::                  &
    hplsupport_BlockCyclicDistArray2dDouble_impl_t
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.private_data)
    ! insert code here (private data members)
    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.private_data)
  end type hplsupport_BlockCyclicDistArray2dDouble_impl_t

contains

  subroutine wrap(obj, self, exception)
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t), target :: obj
    type(hplsupport_BlockCyclicDistArray2dDouble_t) :: self
    type(sidl_BaseInterface_t) :: exception
    external hplsupport_BlockCyclicDistArray2dDouble_wrap
    call hplsupport_BlockCyclicDistArray2dDouble_wrap(obj, self, exception)
  end subroutine wrap

  ! DO-NOT-DELETE splicer.begin(_miscellaneous_code_start)
  ! insert code here (extra code)
  ! DO-NOT-DELETE splicer.end(_miscellaneous_code_start)



  ! 
  ! Method:  _ctor[]
  ! Class constructor called when the class is created.
  ! 

  subroutine ctor_impl(self, exception)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._ctor.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._ctor.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t) :: self
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._ctor)
    ! insert code here (_ctor method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._ctor)
  end subroutine ctor_impl


  ! 
  ! Method:  _dtor[]
  ! Class destructor called when the class is deleted.
  ! 

  subroutine dtor_impl(self, exception)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._dtor.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._dtor.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t) :: self
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._dtor)
    ! insert code here (_dtor method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._dtor)
  end subroutine dtor_impl


  ! 
  ! Method:  _load[]
  ! Static class initializer called exactly once before any user-defined method is dispatched
  ! 

  subroutine load_impl(exception)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._load.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._load.use)
    implicit none
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble._load)
    ! insert code here (_load method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble._load)
  end subroutine load_impl


  ! 
  ! Method:  initData[]
  ! 

  subroutine initData_impl(self, data, exception)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.initData.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.initData.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t) :: self
    type(sidl_opaque_t), intent(in) :: data
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.initData)
    ! insert code here (initData method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.initData)
  end subroutine initData_impl


  ! 
  ! Method:  get[]
  ! 

  function get_impl(self, idx1, idx2, exception) result(retval)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.get.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.get.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t) :: self
    integer (kind=sidl_int), intent(in) :: idx1
    integer (kind=sidl_int), intent(in) :: idx2
    type(sidl_BaseInterface_t), intent(out) :: exception
    !  function result
    real (kind=sidl_double) :: retval

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.get)
    ! insert code here (get method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.get)
  end function get_impl


  ! 
  ! Method:  set[]
  ! 

  subroutine set_impl(self, newVal, idx1, idx2, exception)
    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.set.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.set.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_impl_t) :: self
    real (kind=sidl_double), intent(in) :: newVal
    integer (kind=sidl_int), intent(in) :: idx1
    integer (kind=sidl_int), intent(in) :: idx2
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hplsupport.BlockCyclicDistArray2dDouble.set)
    ! insert code here (set method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hplsupport.BlockCyclicDistArray2dDouble.set)
  end subroutine set_impl


  ! DO-NOT-DELETE splicer.begin(_miscellaneous_code_end)
  ! insert code here (extra code)
  ! DO-NOT-DELETE splicer.end(_miscellaneous_code_end)

end module hplsupport_BlockCyclicDistArray2dDouble_Impl
