_extern proc getOpaqueData(inout inData): opaque;

_extern proc printAddress(opData: opaque);

//pragma "inline"
proc getArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    return a._value.arrayOrdering;  
  }
  return sidl_array_ordering.sidl_row_major_order;
}

//pragma "inline"
proc ensureLocalArray(inout a:[?aDom],
    in mode: sidl_array_ordering = sidl_array_ordering.sidl_row_major_order) var 
    where isRectangularDom(aDom) {
  writeln("ensureLocalArray() starts...");
    
  param arrayRank = a.rank;  
  var arrayOrderMatch = (getArrayOrderMode(a) == mode);
  
  // Create the borrowed domain  
  type locDomType = chpl__buildDomainRuntimeType(defaultBorrowedDistr, arrayRank, aDom._value.idxType, false);
  var locDom: locDomType;
  
  // compute and fill b-array dimension lengths
  var dimRanges: arrayRank * range(aDom._value.idxType, BoundedRangeType.bounded, false);
  if (arrayRank == 1) {
    dimRanges(1) = aDom.low..aDom.high;
    writeln("ensureLocalArray() 1. dimRanges(1)=", dimRanges(1));
  } else {
    for param i in 1..arrayRank do {
      dimRanges(i) = aDom.low(i)..aDom.high(i);  
      writeln("ensureLocalArray() 2. dimRanges(", i, ")=", dimRanges(i));
    }
  }
  locDom._value.dsiSetIndices(dimRanges);
  
  // create the borrowed array
  type locArrType = chpl__buildArrayRuntimeType(locDom, a.eltType);
  var locArr: locArrType;
  
  // borrow data
  if (here.id == aDom.locale.id && (arrayRank == 1 || arrayOrderMatch)) {
    // directly borrow the data for a local array in the correct mode
    var opData: opaque = getOpaqueData(a(aDom.low));
    write("fillRectangularArray() borrow data from (1): ");   
    printAddress(opData);
    locArr._value.borrow(opData); 
  } else {
    // make a local copy of the non-local/distributed array in correct mode
    var rArray = copyRectangularArray(a, !arrayOrderMatch);
    // then borrow over the data
    var opData: opaque = getOpaqueData(rArray(rArray.domain.low));
    write("fillRectangularArray() borrow data from (2): ");     
    printAddress(opData);
    locArr._value.borrow(opData);
  }
  write("ensureLocalArray() locArr data: ");
  printAddress(getOpaqueData(locArr(locArr.domain.low)));
  
  locArr._value.setArrayOrdering(mode);
  
  writeln("ensureLocalArray() input domain: ", aDom);
  writeln("ensureLocalArray() output domain: ", locDom);
  writeln("ensureLocalArray() returns.");
  return locArr;
}

//pragma "inline"
proc syncNonLocalArray(inout src:[], inout target: [?targetDom]) 
    where isRectangularDom(targetDom) {
  writeln("syncNonLocalArray() starts...");  
    
  _extern proc isSameOpaqueData(a: opaque, b: opaque): bool;  
    
  var invertRepresentation = getArrayOrderMode(src) != getArrayOrderMode(target);  
  if (here.id != targetDom.locale.id) {
    writeln("syncNonLocalArray() case-1.");    
    // sync data into non-local array
    fillRectangularArray(src, target, invertRepresentation);
  } else {
    // target is a local array
    var opData1: opaque = getOpaqueData(src(src.domain.low));
    var opData2: opaque = getOpaqueData(target(target.domain.low));
    // If data references are not the same, we need to copy them over
    if (!isSameOpaqueData(opData1, opData2)) { 
      writeln("syncNonLocalArray() case-2.");  
      // local array, but we need to copy back 
      fillRectangularArray(src, target, invertRepresentation);
    }
  }
  writeln("syncNonLocalArray() ends.");  
}

// pragma "inline"
proc copyRectangularArray(in origArr: [],
    in invertRepresentation: bool = false)  var
    where isRectangularDom(origArr.domain) {
   
  writeln("copyRectangularArray() starts...");  
  type indexType = origArr._value.idxType;
  param arrayRank = origArr.rank;
  var origArrDom = origArr.domain;

  var copyDom: domain(rank=arrayRank, idxType=indexType);
  var myRanges : arrayRank * range(indexType);
  if (arrayRank == 1) {
    myRanges(1) = origArrDom.low..origArrDom.high;  
  } else {
    for param i in 1..arrayRank do {
      var loopRange = origArrDom.low(i)..origArrDom.high(i);  
      myRanges(i) = loopRange;
    }
  }
  copyDom._value.dsiSetIndices(myRanges);

  var copyArr: [copyDom] origArr.eltType;
  fillRectangularArray(origArr, copyArr, invertRepresentation);

  writeln("copyRectangularArray() ends.");  
  return copyArr;
}

pragma "inline"
proc fillRectangularArray(in srcArray: [], inout destRectArray: [], 
    in invertRepresentation: bool = false) 
    where isRectangularDom(destRectArray.domain) {
  writeln("fillRectangularArray() starts..."); 
  var srcArrDom = srcArray.domain;
  if (invertRepresentation && srcArray.rank != 1) {
    halt("Array ordering conversion not yet supported!");  
  } else {  
    [i in srcArrDom] destRectArray(i) = srcArray(i);
  }
  writeln("fillRectangularArray() ends."); 
}

proc checkArraysAreEqual(in srcArray: [], inout destArray: []) {
  [i in srcArray.domain] {
    var srcValue = srcArray(i);
    var destValue = destArray(i);
    if (srcValue != destValue) {
      halt("At index ", i, " expected: ", srcValue, ", but found ", destValue);
    }
  }
}