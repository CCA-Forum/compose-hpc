// -*- chpl -*- 
// This fragment will preprocessed to yield sidl.chpl during compile time.

module sidl {
  // FIXME: autogenerate this from sidl.sidl
  _extern record sidl_BaseInterface__object {};

# include <array.inc>

// Start: Borrowed Array related items  
# include <barray.inc>
// End: Borrowed Array related items  

// Start: Distributed Array related items  
// FIXME Enable the include for darray.inc when we can support gasnet compilation
// # include <darray.inc>

pragma "inline"
proc ensureLocalArray(inout a:[?aDom]) var 
    where isRectangularDom(aDom) {

  var res = a;  
  if (here.id != aDom.locale.id) {
    // make a local copy of the array
    res = copyRectangularArray(a);
  } 
  return res;
}

pragma "inline"
proc syncNonLocalArray(in src:[], inout target: [?targetDom]) 
    where isRectangularDom(targetDom) {
  if (here.id != targetDom.locale.id) {
    // sync data into non-local array
    fillRectangularArray(src, target);
  } else {
    // No-op
    target = src;
  }
}

pragma "inline"
proc copyRectangularArray(in a: [])  var
    where isRectangularDom(a.domain) {

  type indexType = a._value.idxType;
  param arrayRank = a.rank;

  var bDom: domain(rank=arrayRank, idxType=indexType);
  var myRanges : arrayRank * range(indexType);
  if (arrayRank == 1) {
    myRanges(1) = a.domain.low..a.domain.high;  
  } else {
    for param i in 1..arrayRank do {
      myRanges(i) = a.domain.low(i)..a.domain.high(i);
    }
  }
  bDom._value.dsiSetIndices(myRanges);

  var b: [bDom] a.eltType;
  [i in bDom] b(i) = a(i); // should we parallelize this?
  return b;
}

pragma "inline"
proc fillRectangularArray(in srcArray: [], inout destRectArray: []) 
    where isRectangularDom(destRectArray.domain) {
  [i in srcArray.domain] destRectArray(i) = srcArray(i);
}
// End: Distributed Array related items  
}
