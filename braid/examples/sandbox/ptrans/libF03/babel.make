IORHDRS = hplsupport_BlockCyclicDistArray2dDouble_IOR.h hplsupport_IOR.h
STUBHDRS = hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h                  \
  hplsupport_BlockCyclicDistArray2dDouble_fStub.h
STUBMODULESRCS = hplsupport_BlockCyclicDistArray2dDouble.F03
STUBSRCS = hplsupport_BlockCyclicDistArray2dDouble_fStub.c
TYPEMODULESRCS = hplsupport_BlockCyclicDistArray2dDouble_type.F03

_deps_hplsupport_BlockCyclicDistArray2dDouble =                               \
  hplsupport_BlockCyclicDistArray2dDouble_type
hplsupport_BlockCyclicDistArray2dDouble$(MOD_SUFFIX) : $(addsuffix $(MOD_SUFFIX), $(_deps_hplsupport_BlockCyclicDistArray2dDouble))

