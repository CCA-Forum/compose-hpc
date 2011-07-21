FSKELSRCS = hpcc_ParallelTranspose_fSkelf.F03
IMPLSRCS = hpcc_ParallelTranspose_Impl.F03 
IORHDRS = hpcc_IOR.h hpcc_ParallelTranspose_IOR.h                             \
  hplsupport_BlockCyclicDistArray2dDouble_IOR.h hplsupport_IOR.h
IORSRCS = hpcc_ParallelTranspose_IOR.c     
SKELSRCS = hpcc_ParallelTranspose_fSkel.c  
STUBHDRS = hpcc_ParallelTranspose_fAbbrev.h hpcc_ParallelTranspose_fStub.h    \
  hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h                           \
  hplsupport_BlockCyclicDistArray2dDouble_fStub.h
STUBMODULESRCS = hpcc_ParallelTranspose.F03                                   \
  hplsupport_BlockCyclicDistArray2dDouble.F03
STUBSRCS = hpcc_ParallelTranspose_fStub.c                                     \
  hplsupport_BlockCyclicDistArray2dDouble_fStub.c
TYPEMODULESRCS = hpcc_ParallelTranspose_type.F03                              \
  hplsupport_BlockCyclicDistArray2dDouble_type.F03

_deps_hpcc_ParallelTranspose =  hpcc_ParallelTranspose_type                   \
  hplsupport_BlockCyclicDistArray2dDouble_type
hpcc_ParallelTranspose$(MOD_SUFFIX) : $(addsuffix $(MOD_SUFFIX), $(_deps_hpcc_ParallelTranspose))

_deps_hpcc_ParallelTranspose_Impl =  hpcc_ParallelTranspose                   \
  hplsupport_BlockCyclicDistArray2dDouble
hpcc_ParallelTranspose_Impl$(MOD_SUFFIX) : $(addsuffix $(MOD_SUFFIX), $(_deps_hpcc_ParallelTranspose_Impl))

_deps_hpcc_ParallelTranspose_fSkelf =  hpcc_ParallelTranspose_Impl            \
  hpcc_ParallelTranspose_type hplsupport_BlockCyclicDistArray2dDouble_type
hpcc_ParallelTranspose_fSkelf$(MOD_SUFFIX) : $(addsuffix $(MOD_SUFFIX), $(_deps_hpcc_ParallelTranspose_fSkelf))

_deps_hplsupport_BlockCyclicDistArray2dDouble =                               \
  hplsupport_BlockCyclicDistArray2dDouble_type
hplsupport_BlockCyclicDistArray2dDouble$(MOD_SUFFIX) : $(addsuffix $(MOD_SUFFIX), $(_deps_hplsupport_BlockCyclicDistArray2dDouble))


