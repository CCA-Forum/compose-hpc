use BlockDist, CyclicDist, BlockCycDist, ReplicatedDist;

///////////////////////////////////////////////////////////////////////////
// Utility functions for distributed arrays and domains
///////////////////////////////////////////////////////////////////////////

/**
 * Utility function to return whether the input domain wraps a 
 * distributed domain. To extend support for custom distr domains
 * a new procedure isDistributedDomainClass(in theDomain: MyCustomDom)
 * will need to be defined.
 */
pragma "inline"
proc isDistributedDomain(d: domain) param {
  // Custom distributions can be supported by adding extra procedures
  // These custom procedures do not need to be defined in this scope
  // Here we only provide support for the standard distributions in chapel
  pragma "inline" proc isDistributedDomainClass(in theDomain: BlockDom) param { return true; }
  pragma "inline" proc isDistributedDomainClass(in theDomain: CyclicDom) param { return true; }
  pragma "inline" proc isDistributedDomainClass(in theDomain: BlockCyclicDom) param { return true; }
  pragma "inline" proc isDistributedDomainClass(in theDomain: ReplicatedDom) param { return true; }
  pragma "inline" proc isDistributedDomainClass(in theDomain) param { return false; }

  return isDistributedDomainClass(d._value);
}

/**
 * Ensure that a local array is returned making a copy if required
 */
pragma "inline"
proc ensureLocalArray(inout a:[?aDom]) var 
    where isRectangularDom(aDom) {

  if (isDistributedDomain(aDom)) {
    // make local copy of the distributed array
    return copyRectangularArray(a);
  } else if (here.id != aDom.locale.id) {
    // make a local copy of the array
    return copyRectangularArray(a);
  } else {
    return a;
  }
}

/**
 * Sync back data into target array if it has non-local elements 
 */
pragma "inline"
proc syncNonLocalArray(in src:[], inout target: [?targetDom]) 
    where isRectangularDom(targetDom) {
  if (isDistributedDomain(targetDom)) {
    // sync data into distr array
    fillRectangularArray(src, target);
  } else if (here.id != targetDom.locale.id) {
    // sync data into non-local array
    fillRectangularArray(src, target);
  } else {
    // No-op
  }
}

/**
 * Returns a copy of a rectangular array with the same domain.
 * If the input array is distributed, it is converted into a 
 * rectangular array.
 */
pragma "inline"
proc copyRectangularArray(in a: [])  var
    where isRectangularDom(a.domain) {

  type indexType = a._value.idxType;
  param arrayRank = a.rank;

  var bDom: domain(rank=arrayRank, idxType=indexType);
  var myRanges : arrayRank * range(indexType);
  for param i in 1..arrayRank do {
    myRanges(i) = a.domain.low(i)..a.domain.high(i);
  }
  bDom._value.dsiSetIndices(myRanges);

  var b: [bDom] a.eltType;
  [i in bDom] b(i) = a(i); // should we parallelize this?
  return b;
}

/**
 * A simple array copy oblivious whether the src/dest arrays are 
 * distributed or not. Only constraint is that the dest array
 * must have a rectangular domain.
 */
pragma "inline"
proc fillRectangularArray(in srcArray: [], inout destRectArray: []) 
    where isRectangularDom(destRectArray.domain) {
  [i in srcArray.domain] destRectArray(i) = srcArray(i);
}

///////////////////////////////////////////////////////////////////////////
// Example use of the methods:
///////////////////////////////////////////////////////////////////////////
// 
// config const n = 4;
// const Space = [0.. #n, 0.. #n];
// const BlockSpace = Space dmapped Block(boundingBox=Space);
// 
// var copyBA = copyRectangularArray(BA);
// [i in 0.. #n] copyBA(i, i) = 10 * i;
// fillRectangularArray(copyBA, BA);
// writeln(BA);
// 
// isDistributedDomain(BA.domain); // true
// isDistributedDomain(copyBA.domain); // false
// 
///////////////////////////////////////////////////////////////////////////

