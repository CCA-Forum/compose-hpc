use DSIUtil;
use DefaultRectangular;

///////////////////////////////////////////////////////////////////////////
// The generic borrowed data class
///////////////////////////////////////////////////////////////////////////

// dynamic data block class
pragma "data class"
class _borrowedData {
  type eltType;
  var opData: opaque;

  proc ~_borrowedData() {
    // Nothing to free, no data owned!
  }
  pragma "inline" proc init(opaqueData: opaque) {
    // Just copy over the reference to the data
    this.opData = opaqueData;
    __primitive("ref_borrow", this, "_data", opaqueData);
  }
  pragma "inline" proc this(i: integral) var {
    // rely on chapel compiler to generate lvalue
    return __primitive("array_get", this, i);
  }
}

/**
 * Sample signatures of methods to allocate memory externally.
 *
 * _extern proc allocateData(typeSize: int(32), numElements: int(32)): opaque;
 * 
 * _extern proc deallocateData(bData: opaque);
 * 
 * _extern proc getBorrowedData(): opaque;
 * 
 */

///////////////////////////////////////////////////////////////////////////
// BorrowedDistribution class to create the BorrowedDomain
///////////////////////////////////////////////////////////////////////////

class BorrowedDist: BaseDist {

  proc dsiClone() return this;

  proc dsiAssign(other: this.type) { }

  proc dsiNewRectangularDom(param rank: int, type idxType, param stridable: bool)
    return new BorrowedRectangularDom(rank, idxType, stridable, this);
}

///////////////////////////////////////////////////////////////////////////
// BorrowedRectangular domain to create the borrowed array
// delegates almost all its methods to the standard DefaultRectangularDom 
// Main method that has changed:
// - dsiBuildArray()
// - ranges()
///////////////////////////////////////////////////////////////////////////

class BorrowedRectangularDom: BaseRectangularDom {
  param rank : int;
  type idxType;
  param stridable: bool;
  
  var dist: BorrowedDist;
  // var ranges: rank*range(idxType,BoundedRangeType.bounded,stridable);
  var rectDom: DefaultRectangularDom(rank, idxType, stridable);

  proc linksDistribution() param return false;
  proc dsiLinksDistribution()     return false;

  proc BorrowedRectangularDom(param rank, type idxType, param stridable, dist) {
    this.dist = dist;
    this.rectDom = new DefaultRectangularDom(rank, idxType, stridable, defaultDist._value);
  }

  proc initIndices(theLimits ...?k) {
    var myRanges : k * range(idxType, BoundedRangeType.bounded, stridable);
    for param i in 1..k do {
      myRanges(i) = 0..(theLimits(i) - 1);
    }
    dsiSetIndices(myRanges);
  }

  proc ranges(idx) {
    return rectDom.ranges(idx);
  }

  proc dsiClear() { this.rectDom.dsiClear(); }
  
  // function and iterator versions, also for setIndices
  proc dsiGetIndices() { return this.rectDom.dsiGetIndices(); }

  proc dsiSetIndices(x) { this.rectDom.dsiSetIndices(x); }

  iter these_help(param d: int) {
    for i in this.rectDom.these_help(d) do
      yield i;
  }

  iter these_help(param d: int, block) {
    for i in this.rectDom.these_help(d, block) do
      yield i;
  }

  iter these() {
    for i in this.rectDom.these() do
      yield i;
  }

  iter these(param tag: iterator) where tag == iterator.leader {
    for i in this.rectDom.these(tag) do
      yield i;
  }

  iter these(param tag: iterator, follower) where tag == iterator.follower {
    for i in this.rectDom.these(tag, follower) do
      yield i;
  }

  proc dsiMember(ind: rank*idxType) { return this.rectDom.dsiMember(ind); }

  proc dsiIndexOrder(ind: rank*idxType) { return this.rectDom.dsiIndexOrder(ind); }

  proc dsiDims() { return this.rectDom.dsiDims(); }

  proc dsiDim(d : int) { return this.rectDom.dsiDim(d); }

  // optional, is this necesary? probably not now that
  // homogeneous tuples are implemented as C vectors.
  proc dsiDim(param d : int) { return this.rectDom.dsiDim(d); }

  proc dsiNumIndices { return this.rectDom.dsiNumIndices; }

  proc dsiLow { return this.rectDom.dsiLow; }

  proc dsiHigh { return this.rectDom.dsiHigh; }

  proc dsiAlignedLow { return this.rectDom.dsiAlignedLow; }

  proc dsiAlignedHigh { return this.rectDom.dsiAlignedHigh; }

  proc dsiStride { return this.rectDom.dsiStride; }

  proc dsiAlignment { return this.rectDom.dsiAlignment; }

  proc dsiFirst { return this.rectDom.dsiFirst; }

  proc dsiLast { return this.rectDom.dsiLast; }

  proc dsiBuildArray(type eltType) {
    return new BorrowedRectangularArr(eltType=eltType, rank=rank, idxType=idxType,
                                    stridable=stridable, dom=this);
  }

  proc dsiBuildRectangularDom(param rank: int, type idxType, param stridable: bool,
        ranges: rank*range(idxType, BoundedRangeType.bounded, stridable)) {
    return this.rectDom.dsiBuildRectangularDom(rank, idxType, stridable, ranges);
  }
}

///////////////////////////////////////////////////////////////////////////
// BorrowedRectangular array that can refer to externally allocated memory 
// based on DefaultRectangularArr, notable method changes are:
// - this()
// - dsiDestroyData()
// - initialize()
// - dsiReindex()
// - dsiSlice()
// - dsiRankChange()
// - dsiReallocate()
// - dsiLocalSlice()
///////////////////////////////////////////////////////////////////////////

class BorrowedRectangularArr: BaseArr {

  type eltType;
  param rank : int;
  type idxType;
  param stridable: bool;

  var dom : BorrowedRectangularDom(rank=rank, idxType=idxType, stridable=stridable);
  var off: rank*idxType;
  var blk: rank*idxType;
  var str: rank*chpl__signedType(idxType);
  var origin: idxType;
  var factoredOffs: idxType;
  var bData : _borrowedData(eltType);
  var noinit: bool = false;

  proc canCopyFromDevice param return true;

  // end class definition here, then defined secondary methods below

  // can the compiler create this automatically?
  proc dsiGetBaseDom() { return dom; }

  proc dsiDestroyData() {
    // Nothing to do
  }

  iter these() var {
    if rank == 1 {
      // This is specialized to avoid overheads of calling dsiAccess()
      if !dom.stridable {
        // This is specialized because the strided version disables the
        // "single loop iterator" optimization
        var first = getDataIndex(dom.dsiLow);
        var second = getDataIndex(dom.dsiLow+dom.ranges(1).stride:idxType);
        var step = (second-first):chpl__signedType(idxType);
        var last = first + (dom.dsiNumIndices-1) * step:idxType;
        for i in first..last by step do
          yield bData(i);
      } else {
        const stride = dom.ranges(1).stride: idxType,
              start  = dom.ranges(1).first,
              first  = getDataIndex(start),
              second = getDataIndex(start + stride),
              step   = (second-first):chpl__signedType(idxType),
              last   = first + (dom.ranges(1).length-1) * step:idxType;
        if step > 0 then
          for i in first..last by step do
            yield bData(i);
        else
          for i in last..first by step do
            yield bData(i);
      }
    } else {
      for i in dom do
        yield dsiAccess(i);
    }
  }

  iter these(param tag: iterator) where tag == iterator.leader {
    for follower in dom.these(tag) do
      yield follower;
  }

  iter these(param tag: iterator, follower) var where tag == iterator.follower {
    if debugDefaultDist then
      writeln("*** In array follower code:"); // [\n", this, "]");
    for i in dom.these(tag=iterator.follower, follower) {
      __primitive("noalias pragma");
      yield dsiAccess(i);
    }
  }

  proc computeFactoredOffs() {
    factoredOffs = 0:idxType;
    for param i in 1..rank do {
      factoredOffs = factoredOffs + blk(i) * off(i);
    }
  }
  
  // change name to setup and call after constructor call sites
  // we want to get rid of all initialize functions everywhere
  proc initialize() {
    if noinit == true then return;
    for param dim in 1..rank {
      off(dim) = dom.dsiDim(dim).alignedLow;
      str(dim) = dom.dsiDim(dim).stride;
    }
    blk(rank) = 1:idxType;
    for param dim in 1..rank-1 by -1 do
      blk(dim) = blk(dim+1) * dom.dsiDim(dim+1).length;
    computeFactoredOffs();
    var size = blk(1) * dom.dsiDim(1).length;
    // Do not initialize data here, user will explicitly init the data
  }

  proc borrow(opData: opaque) { 
    this.bData = new _borrowedData(eltType);
    this.bData.init(opData); 
  }

  pragma "inline"
  proc getDataIndex(ind: idxType ...1) where rank == 1 {
    return getDataIndex(ind);
  }

  pragma "inline"
  proc getDataIndex(ind: rank * idxType) {
    var sum = origin;
    if stridable {
      for param i in 1..rank do
        sum += (ind(i) - off(i)) * blk(i) / abs(str(i)):idxType;
    } else {
      for param i in 1..rank do
        sum += ind(i) * blk(i);
      sum -= factoredOffs;
    }
    return sum;
  }

  proc this(ind: idxType ...?numItems) var where rank == numItems {
    var indTuple: numItems * idxType;
    for param i in 1..numItems do {
      indTuple(i) = ind(i);
    }
    return dsiAccess(indTuple);
  }

  proc this(ind: rank*idxType) var {
    return dsiAccess(ind);
  }

  // only need second version because wrapper record can pass a 1-tuple
  pragma "inline"
  proc dsiAccess(ind: idxType ...1) var where rank == 1 {
    return dsiAccess(ind);
  }

  pragma "inline"
  proc dsiAccess(ind : rank*idxType) var {
    if boundsChecking then
      if !dom.dsiMember(ind) then
        halt("array index out of bounds: ", ind);
    var dataInd = getDataIndex(ind);
    //assert(dataInd >= 0);
    //assert(numelm >= 0); // ensure it has been initialized
    //assert(dataInd: uint(64) < numelm: uint(64));
    return bData(dataInd);
  }

  proc dsiReindex(d: DefaultRectangularDom) {
    halt("dsiReindex() not supported for BorrowedRectangularArray");
  }

  proc dsiSlice(d: DefaultRectangularDom) {
    halt("dsiSlice() not supported for BorrowedRectangularArray");
  }

  proc dsiRankChange(d, param newRank: int, param newStridable: bool, args) {
    halt("dsiRankChange() not supported for BorrowedRectangularArray");
  }

  proc dsiReallocate(d: domain) {
    halt("dsiReallocate() not supported for BorrowedRectangularArray");
  }

  proc dsiLocalSlice(ranges) {
    halt("all dsiLocalSlice calls on DefaultRectangulars should be handled in ChapelArray.chpl");
  }
}

proc BorrowedRectangularDom.dsiSerialWrite(f: Writer) {
  f.write("[", dsiDim(1));
  for i in 2..rank do
    f.write(", ", dsiDim(i));
  f.write("]");
}

proc BorrowedRectangularArr.dsiSerialWrite(f: Writer) {
  proc recursiveArrayWriter(in idx: rank*idxType, dim=1, in last=false) {
    var makeStridePositive = if dom.ranges(dim).stride > 0 then 1 else -1;
    if dim == rank {
      var first = true;
      if debugDefaultDist then f.writeln(dom.ranges(dim));
      for j in dom.ranges(dim) by makeStridePositive {
        if first then first = false; else f.write(" ");
        idx(dim) = j;
        f.write(dsiAccess(idx));
      }
    } else {
      for j in dom.ranges(dim) by makeStridePositive {
        var lastIdx =  dom.ranges(dim).last;
        idx(dim) = j;
        recursiveArrayWriter(idx, dim=dim+1,
                             last=(last || dim == 1) && (j == lastIdx));
      }
    }
    if !last && dim != 1 then
      f.writeln();
  }
  const zeroTup: rank*idxType;
  recursiveArrayWriter(zeroTup);
}


///////////////////////////////////////////////////////////////////////////
// Utility functions to create borrowed arrays
///////////////////////////////////////////////////////////////////////////

var defaultBorrowedDistr = _newDistribution(new BorrowedDist());

// Define custom copy method for borrowed arrays
proc chpl__initCopy(a: []) where 
    a._dom._value.type == BorrowedRectangularDom(a._value.rank, a._value.idxType, a._value.stridable) {

  var b : [a._dom] a.eltType;
  b._value.borrow(a._value.bData.opData);

  return b;
}

pragma "inline" proc createBorrowedArray(type arrayIndexType, type arrayElmntType, 
        bData: opaque, arraySize: int(32)...?arrayRank) {
  
  type locDomType = chpl__buildDomainRuntimeType(defaultBorrowedDistr, arrayRank, arrayIndexType, false);
  var locDom: locDomType;
  locDom._value.initIndices((...arraySize));
  
  type locArrType = chpl__buildArrayRuntimeType(locDom, arrayElmntType);
  var locArr: locArrType;  
  locArr._value.borrow(bData);

  return locArr;
}

pragma "inline" proc createBorrowedArray1d(sa: sidl.Array) {
  if (sa.dim() != 1) {
    halt("input array is not of rank 1");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0));  
}
pragma "inline" proc createBorrowedArray2d(sa: sidl.Array) {
  if (sa.dim() != 2) {
    halt("input array is not of rank 2");  
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1));  
}
pragma "inline" proc createBorrowedArray3d(sa: sidl.Array) {
  if (sa.dim() != 3) {
    halt("input array is not of rank 3");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1), sa.length(2));  
}
pragma "inline" proc createBorrowedArray4d(sa: sidl.Array) {
  if (sa.dim() != 4) {
    halt("input array is not of rank 4");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1), sa.length(2), sa.length(3));  
}
pragma "inline" proc createBorrowedArray5d(sa: sidl.Array) {
  if (sa.dim() != 5) {
    halt("input array is not of rank 5");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1), sa.length(2), sa.length(3), 
            sa.length(4));  
}
pragma "inline" proc createBorrowedArray6d(sa: sidl.Array) {
  if (sa.dim() != 6) {
    halt("input array is not of rank 6");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1), sa.length(2), sa.length(3), 
            sa.length(4), sa.length(5));  
}
pragma "inline" proc createBorrowedArray7d(sa: sidl.Array) {
  if (sa.dim() != 7) {
    halt("input array is not of rank 7");
  }
  return createBorrowedArray(int(32), sa.ScalarType, sa.first(), 
            sa.length(0), sa.length(1), sa.length(2), sa.length(3), 
            sa.length(4), sa.length(5), sa.length(6));  
}

proc resetBorrowedArray(bArr: BorrowedRectangularArr, bData: opaque, 
        arraySize ...?arrayRank) where bArr.rank == arrayRank {

  var bDom = bArr.getBaseDom();
  bDom.initIndices((...arraySize));
  bArr.borrow(bData);
  return bArr;
}

///////////////////////////////////////////////////////////////////////////
// Start example use of borrowed array
///////////////////////////////////////////////////////////////////////////
// 
// _extern proc allocateData(typeSize: int(32), numElements: int(32)): opaque;
// 
// type arrayIndexType = int(32);
// type arrayElmntType = real(64);
// var arraySize1d = 10;
// 
// var bData1d: opaque;
// local { bData1d = allocateData(numBits(arrayElmntType), arraySize1d); }
// 
// var bArr1d = createBorrowedArray(arrayIndexType, arrayElmntType, bData1d, arraySize1d);
// [i in 0.. #arraySize1d by 2] { bArr1d(i) = i; }
// [i in 0.. #arraySize1d] { writeln("bArr1d(", i, ") = ", bArr1d(i)); }
// 
// var arraySize2di = 3;
// var arraySize2dj = 5;
// 
// var bData2d: opaque;
// local { bData2d = allocateData(numBits(arrayElmntType), arraySize2di * arraySize2dj); }
// 
// var bArr2d = createBorrowedArray(arrayIndexType, arrayElmntType, bData2d, arraySize2di, arraySize2dj);
// [(i, j) in [0.. #arraySize2di, 0.. #arraySize2dj by 2]] { bArr2d(i, j) = (10 * i) + j; }
// [(i, j) in [0.. #arraySize2di, 0.. #arraySize2dj]] { writeln("bArr2d(", i, ", ", j, ") = ", bArr2d(i, j)); }
// 
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

