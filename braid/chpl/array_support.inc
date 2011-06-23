// TODO Remove all debug prints when done

_extern proc getOpaqueData(inout inData): opaque;

_extern proc printAddress(opData: opaque);

param debug = false;

//pragma "inline"
proc getArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    return a._value.arrayOrdering;
  }
  // Default ordering for chapel arrays is row-major
  return sidl_array_ordering.sidl_row_major_order;
}

//pragma "inline"
proc getSidlArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    // Borrowed arrays have inherited their mode
    return a._value.arrayOrdering;
  }
  // Default ordering for rarrays is column-major
  return sidl_array_ordering.sidl_column_major_order;
}

//pragma "inline"
proc ensureLocalArray(inout a:[?aDom], aData: opaque) var
    where isRectangularDom(aDom) {
  if (debug) { writeln("ensureLocalArray() starts..."); }

  param arrayRank = a.rank;
  var arrayOrder = getSidlArrayOrderMode(a);
  var arrayOrderMatch = (getArrayOrderMode(a) == arrayOrder);

  // Create the borrowed domain
  type locDomType = chpl__buildDomainRuntimeType(defaultBorrowedDistr, arrayRank, aDom._value.idxType, false);
  var locDom: locDomType;

  // compute and fill b-array dimension lengths
  var dimRanges: arrayRank * range(aDom._value.idxType, BoundedRangeType.bounded, false);
  if (arrayRank == 1) {
    dimRanges(1) = aDom.low..aDom.high;
    if (debug) { writeln("ensureLocalArray() 1. dimRanges(1)=", dimRanges(1)); }
  } else {
    for param i in 1..arrayRank do {
      dimRanges(i) = aDom.low(i)..aDom.high(i);
      if (debug) { writeln("ensureLocalArray() 2. dimRanges(", i, ")=", dimRanges(i)); }
    }
  }
  locDom._value.dsiSetIndices(dimRanges);

  // create the borrowed array in expected sidl mode
  type locArrType = chpl__buildArrayRuntimeType(locDom, a.eltType);
  var locArr: locArrType;
  locArr._value.setArrayOrdering(arrayOrder);
  
  // borrow data
  if (here.id == aDom.locale.id && (arrayRank == 1 || arrayOrderMatch)) {
    // directly borrow the data for a local array in the correct mode
    var opData: opaque = aData;
    if (debug) { write("fillRectangularArray() borrow data from (1): "); }
    if (debug) { printAddress(opData); }
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(false);
  } else {
    // make a local copy of the non-local/distributed array in correct mode
    // self allocate the data, set as owner and then make element-wise copy
    var opData: opaque = allocateData(numBits(locArr.eltType), a.numElements);
    if (debug) { write("fillRectangularArray() borrow data from (2): "); }
    if (debug) { printAddress(opData); }
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(true);
    [i in aDom] locArr(i) = a(i);
  }
  if (debug) { write("ensureLocalArray() locArr data: "); }
  if (debug) { printAddress(getOpaqueData(locArr(locArr.domain.low))); }

  if (debug) { writeln("ensureLocalArray() input domain: ", aDom); }
  if (debug) { writeln("ensureLocalArray() output domain: ", locDom); }
  if (debug) { writeln("ensureLocalArray() returns."); }
  return locArr;
}

//pragma "inline"
proc syncNonLocalArray(inout src:[], inout target: [?targetDom])
    where isRectangularDom(targetDom) {
  if (true || debug) { writeln("syncNonLocalArray() starts..."); }

  if (true || debug) { writeln("syncNonLocalArray.1.src: ", typeToString(src.type)); writeln(src); }
  if (true || debug) { writeln("syncNonLocalArray.1.target: ", typeToString(target.type)); writeln(target); }

  _extern proc isSameOpaqueData(a: opaque, b: opaque): bool;

  var arrayCopyReqd = false;
  if (here.id != targetDom.locale.id) {
    if (true || debug) { writeln("syncNonLocalArray() case-1."); }
    arrayCopyReqd = true;
  } else if (getArrayOrderMode(src) != getArrayOrderMode(target)) {
    if (true || debug) { writeln("syncNonLocalArray() case-2."); }
    arrayCopyReqd = true;
  } else {
    // target is a local array
    var opData1: opaque = getOpaqueData(src(src.domain.low));
    var opData2: opaque = getOpaqueData(target(target.domain.low));
    // If data references are not the same, we need to copy them over
    if (!isSameOpaqueData(opData1, opData2)) {
      if (true || debug) { writeln("syncNonLocalArray() case-3."); }
      arrayCopyReqd = true;
    }
  }
  if (arrayCopyReqd) {
    [i in src.domain] target(i) = src(i);
  }
  if (true || debug) { writeln("syncNonLocalArray.2.target: "); writeln(target); }
  if (true || debug) { writeln("syncNonLocalArray() ends."); }
}

proc checkArraysAreEqual(in srcArray: [], inout destArray: []) {
  [i in srcArray.domain] {
    var srcValue = srcArray(i);
    var destValue = destArray(i);
    if (srcValue != destValue) {
      writeln("ERROR: At index ", i, " expected: ", srcValue, ", but found ", destValue);
    }
  }
}

proc computeLowerUpperAndStride(in srcArray: [?srcDom]) {

  param arrayRank = srcArray.rank;
  var result: [0..2][1..arrayRank] int(32);
  var arrayOrderMode = getArrayOrderMode(srcArray);

  for i in [1..arrayRank] {
    var r: range = srcDom.dim(i);
    result[0][i] = r.low;
    result[1][i] = r.high;
  }

  if (arrayOrderMode == sidl_array_ordering.sidl_column_major_order) {
    var loopStride = 1;
    for i in [1..arrayRank] {
      result[2][i] = loopStride;
      var rs: range = srcDom.dim(i);
      loopStride = loopStride * (rs.high - rs.low + 1);
    }
  } else {
    var loopStride = 1;
    for i in [1..arrayRank] {
      var stride_idx = arrayRank - i + 1;
      result[2][stride_idx] = loopStride;
      var rs: range = srcDom.dim(stride_idx);
      loopStride = loopStride * (rs.high - rs.low + 1);
    }
  }

  return result;
}