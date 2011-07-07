
proc impl_hplsupport_SimpleArray1dInt_getFromArray_chpl(arrayWrapper, idx1: int(32)) {
  return arrayWrapper.get(idx1);
}

proc impl_hplsupport_SimpleArray1dInt_setIntoArray_chpl(arrayWrapper, newVal: int(32), idx1: int(32)) {
  arrayWrapper.set(newVal, idx1);
}