
proc impl_hplsupport_BlockCyclicDistArray2dDouble_getFromArray_chpl(anArray, idx1: int(32), idx2: int(32)) {
  return anArray(idx1, idx2);
}

proc impl_hplsupport_BlockCyclicDistArray2dDouble_setIntoArray_chpl(anArray, newVal: real(64), idx1: int(32), idx2: int(32)) {
  anArray(idx1, idx2) = newVal;
}