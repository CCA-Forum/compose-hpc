
proc impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(arrayWrapper, idx1: int(32), idx2: int(32)) {
  return arrayWrapper.get(idx1, idx2);
}

proc impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(arrayWrapper, newVal: real(64), idx1: int(32), idx2: int(32)) {
  arrayWrapper.set(newVal, idx1, idx2);
}