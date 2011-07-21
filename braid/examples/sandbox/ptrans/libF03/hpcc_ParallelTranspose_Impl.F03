! 
! File:          hpcc_ParallelTranspose_Impl.F03
! Symbol:        hpcc.ParallelTranspose-v0.1
! Symbol Type:   class
! Babel Version: 2.0.0 (Revision: 7138  trunk)
! Description:   Server-side implementation for hpcc.ParallelTranspose
! 
! WARNING: Automatically generated; only changes within splicers preserved
! 
! 


! 
! Symbol "hpcc.ParallelTranspose" (version 0.1)
! 


#include "sidl_BaseException_fAbbrev.h"
#include "hpcc_ParallelTranspose_fAbbrev.h"
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

module hpcc_ParallelTranspose_Impl
  use sidl
  use sidl_BaseInterface
  use sidl_RuntimeException
  use hpcc_ParallelTranspose
  use hplsupport_BlockCyclicDistArray2dDouble
  ! DO-NOT-DELETE splicer.begin(module.use)
  ! 
  ! Add use statements here
  ! 

  ! DO-NOT-DELETE splicer.end(module.use)


  type, extends(hpcc_ParallelTranspose_t) :: hpcc_ParallelTranspose_impl_t
    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose.private_data)
    ! insert code here (private data members)
    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose.private_data)
  end type hpcc_ParallelTranspose_impl_t

contains

  subroutine wrap(obj, self, exception)
    type(hpcc_ParallelTranspose_impl_t), target :: obj
    type(hpcc_ParallelTranspose_t) :: self
    type(sidl_BaseInterface_t) :: exception
    external hpcc_ParallelTranspose_wrap
    call hpcc_ParallelTranspose_wrap(obj, self, exception)
  end subroutine wrap

  ! DO-NOT-DELETE splicer.begin(_miscellaneous_code_start)
  ! insert code here (extra code)
  ! DO-NOT-DELETE splicer.end(_miscellaneous_code_start)



  ! 
  ! Method:  _ctor[]
  ! Class constructor called when the class is created.
  ! 

  subroutine ctor_impl(self, exception)
    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._ctor.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._ctor.use)
    implicit none
    type(hpcc_ParallelTranspose_impl_t) :: self
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._ctor)
    ! insert code here (_ctor method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._ctor)
  end subroutine ctor_impl


  ! 
  ! Method:  _dtor[]
  ! Class destructor called when the class is deleted.
  ! 

  subroutine dtor_impl(self, exception)
    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._dtor.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._dtor.use)
    implicit none
    type(hpcc_ParallelTranspose_impl_t) :: self
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._dtor)
    ! insert code here (_dtor method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._dtor)
  end subroutine dtor_impl


  ! 
  ! Method:  _load[]
  ! Static class initializer called exactly once before any user-defined method is dispatched
  ! 

  subroutine load_impl(exception)
    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._load.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._load.use)
    implicit none
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose._load)
    ! insert code here (_load method)
    ! 
    ! This method has not been implemented
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose._load)
  end subroutine load_impl


  ! 
  ! Method:  ptransCompute[]
  ! 

  subroutine ptransCompute_impl(a, c, beta, i, j, exception)
    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose.ptransCompute.use)
    ! 
    ! Add use statements here
    ! 

    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose.ptransCompute.use)
    implicit none
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: a
    type(hplsupport_BlockCyclicDistArray2dDouble_t), intent(in) :: c
    real (kind=sidl_double), intent(in) :: beta
    integer (kind=sidl_int), intent(in) :: i
    integer (kind=sidl_int), intent(in) :: j
    type(sidl_BaseInterface_t), intent(out) :: exception

    ! DO-NOT-DELETE splicer.begin(hpcc.ParallelTranspose.ptransCompute)
    ! insert code here (ptransCompute method)
     
    write (*,*) "Inside ptransCompute_impl()"
    
    ! DO-NOT-DELETE splicer.end(hpcc.ParallelTranspose.ptransCompute)
  end subroutine ptransCompute_impl


  ! DO-NOT-DELETE splicer.begin(_miscellaneous_code_end)
  ! insert code here (extra code)
  ! DO-NOT-DELETE splicer.end(_miscellaneous_code_end)

end module hpcc_ParallelTranspose_Impl
