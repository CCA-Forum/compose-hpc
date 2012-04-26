// -*- chpl -*- This fragment will be included in sidl.chpl during compile time.
// TODO Remove all debug prints when done

extern proc getOpaqueData(inout inData): opaque;

proc performSanityCheck(aDom: domain, varName: string) {
  if (!isRectangularDom(aDom)) {
    halt(varName, ".domain is not rectangular");
  }
  if (aDom._value.stridable) {
    halt(varName, ".domain is stridable. Stridable domains are not supported.");
  }
}

proc getArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    return a._value.arrayOrdering;
  }
  // Default ordering for chapel arrays is row-major
  return sidl_array_ordering.sidl_row_major_order;
}

proc getSidlArrayOrderMode(in a: [?aDom]): sidl_array_ordering {
  var aDomain = aDom._value;
  if (aDomain.type == BorrowedRectangularDom(aDom.rank, aDomain.idxType, aDomain.stridable)) {
    // Borrowed arrays have inherited their mode
    return a._value.arrayOrdering;
  }
  // Default ordering for rarrays is column-major
  return sidl_array_ordering.sidl_column_major_order;
}

proc ensureLocalArray(inout a:[?aDom], aData: opaque) var
    where isRectangularDom(aDom) {
  
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
  } else {
    for param i in 1..arrayRank do {
      dimRanges(i) = aDom.low(i)..aDom.high(i);
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
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(false);
  } else {
    // make a local copy of the non-local/distributed array in correct
    // order (we expect column-major)
    // self allocate the data, set as owner and then make element-wise copy
    var opData: opaque = allocateData(numBits(locArr.eltType), a.numElements: int(32));
    locArr._value.borrow(opData);
    locArr._value.setDataOwner(true);
    [i in aDom] locArr(i) = a(i);
  }
  return locArr;
}

proc syncNonLocalArray(inout src:[], inout target: [?targetDom])
    where isRectangularDom(targetDom) {

  extern proc isSameOpaqueData(a: opaque, b: opaque): bool;

  var arrayCopyReqd = false;
  if (here.id != targetDom.locale.id) {
    arrayCopyReqd = true;
  } else if (getArrayOrderMode(src) != getArrayOrderMode(target)) {
    arrayCopyReqd = true;
  } else {
    // target is a local array
    var opData1: opaque = getOpaqueData(src(src.domain.low));
    var opData2: opaque = getOpaqueData(target(target.domain.low));
    // If data references are not the same, we need to copy them over
    if (!isSameOpaqueData(opData1, opData2)) {
      arrayCopyReqd = true;
    }
  }
  if (arrayCopyReqd) {
    [i in targetDom] target(i) = src(i);
  }
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
  var result: [0..2][1..arrayRank] int(64);
  var arrayOrderMode = getArrayOrderMode(srcArray);

  for i in [1..arrayRank] {
    var r: range = srcDom.dim(i);
    result[0][i] = r.low;
    result[1][i] = r.high;
  }
  // rely on the blk value from the array to provide the strides
  result[2] = srcArray._value.blk;
  
  return result;
}