// 
// File:        arraytests.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Simple test on the ArrayTest static methods
// 
use ArrayTest;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

var part_no: int = 0;
var sidl_ex: SidlBaseException = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);
var magic_number = 13;

proc init_part()
{
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part " + part_no, sidl_ex);
}

proc run_part(msg: string, result: bool)
{
  var r: ResultType;
  tracker.writeComment(msg, sidl_ex);
  if (result) then
    r = ResultType.PASS;
  else 
    r = ResultType.FAIL;
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("End of part " + part_no, sidl_ex);
}

/**
 * Fill the stack with random junk.
 */
proc clearstack(magicNumber: int): int
{
//  var chunk: 2048*int;
//  for(i = 0; i < 2048; i++){
//    chunk[i] = rand() + magicNumber;
//  }
//  for(i = 0; i < 16; i++){
//    magicNumber += chunk[rand() & 2047];
//  }
  return magicNumber;
}


var TEST_SIZE = 345; /* size of one dimensional arrays */
var TEST_DIM1 = 17; /* first dimension of 2-d arrays */
var TEST_DIM2 = 13; /* second dimension of 2-d arrays */

//synch::ResultType result = synch::ResultType_PASS;
var magicNumber = 13;
var obj = new ArrayTest.ArrayOps();
tracker.setExpectations(-1, sidl_ex);
  
//tracker.setExpectations(158);

  /* { */
  /*   const int32_t numElem[] = { TEST_SIZE/2 }; */
  /*   const int32_t start[] = { 0 }; */
  /*   const int32_t stride[] = { 2 }; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray = ArrayTest.ArrayOps.createBool(TEST_SIZE); */
  /*   init_part(); run_part("createBool", barray._not_nil()); */
  /*   init_part(); run_part("createBool", ArrayTest.ArrayOps.checkBool(barray, sidl_ex) == true); */
  /*   init_part(); run_part("createBool", ArrayTest.ArrayOps.reverseBool(barray, true) == true); */
  /*   sidl::array<bool> sliced = barray.slice(1, numElem, start, stride); */
  /*   sidl::array<bool> cpy; */
  /*   init_part(); run_part("createBool", sliced._not_nil()); */
  /*   sliced.smartCopy(); */
  /*   init_part(); run_part("createBool", sliced._not_nil()); */
  /*   cpy = barray; */
  /*   cpy.smartCopy(); */
  /*   init_part(); run_part("createBool", cpy._not_nil()); */
  /* } */

{
  magicNumber = clearstack(magicNumber);
  var barray: sidl.Array(bool, sidl_bool__array);

  ArrayTest.ArrayOps_static.makeBool(218, barray, sidl_ex);
  init_part(); run_part("makeBool218", ArrayTest.ArrayOps_static.checkBool(barray, sidl_ex) == true);
  init_part(); run_part("makeBool218", ArrayTest.ArrayOps_static.reverseBool(barray, false, sidl_ex) == true);
  init_part(); run_part("makeBool218", ArrayTest.ArrayOps_static.checkBool(barray, sidl_ex) == false);
  barray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
  
{
  magicNumber = clearstack(magicNumber);
  var barray: sidl.Array(bool, sidl_bool__array);
  ArrayTest.ArrayOps_static.makeBool(9, barray, sidl_ex);
  init_part(); run_part("makeBool9", ArrayTest.ArrayOps_static.reverseBool(barray, false, sidl_ex) == true);
  init_part(); run_part("makeBool9", ArrayTest.ArrayOps_static.checkBool(barray, sidl_ex) == true);
  barray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
  
{
  magicNumber = clearstack(magicNumber);
  var carray: sidl.Array(string, sidl_char__array) = ArrayTest.ArrayOps_static.createChar(TEST_SIZE, sidl_ex);
  init_part(); run_part("createChar", carray._not_nil());
  init_part(); run_part("createChar", ArrayTest.ArrayOps_static.checkChar(carray, sidl_ex) == true);
  init_part(); run_part("createChar", ArrayTest.ArrayOps_static.reverseChar(carray, true, sidl_ex) == true);
  carray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var carray: sidl.Array(string, sidl_char__array);
  ArrayTest.ArrayOps_static.makeChar(218, carray, sidl_ex);
  init_part(); run_part("makeChar", ArrayTest.ArrayOps_static.checkChar(carray, sidl_ex) == true);
  init_part(); run_part("makeChar", ArrayTest.ArrayOps_static.reverseChar(carray, false, sidl_ex) == true);
  init_part(); run_part("makeChar", ArrayTest.ArrayOps_static.checkChar(carray, sidl_ex) == false);
  carray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array) = ArrayTest.ArrayOps_static.createInt(TEST_SIZE, sidl_ex);
  init_part(); run_part("createInt", iarray._not_nil());
  init_part(); run_part("createInt", ArrayTest.ArrayOps_static.checkInt(iarray, sidl_ex) == true);
  init_part(); run_part("createInt", ArrayTest.ArrayOps_static.reverseInt(iarray, true, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  ArrayTest.ArrayOps_static.makeInt(218, iarray, sidl_ex);
  init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.checkInt(iarray, sidl_ex) == true);
  init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.reverseInt(iarray, false, sidl_ex) == true);
  init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.checkInt(iarray, sidl_ex) == false);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  tracker.writeComment("Start: Check sidl.borrow_int_Array", sidl_ex);

  magicNumber = clearstack(magicNumber);
  var borrowed: sidl.Array(int(32), sidl_int__array);
  var elements: [0..31] int(32) =
    (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
     89, 97, 101, 103, 107, 109, 113, 127, 131);

  borrowed = sidl.borrow_int_Array(elements, int_ptr(elements[0]));
  init_part(); run_part("borrowed_int: not-nil", borrowed._not_nil());

  var resCheckInt1 = ArrayTest.ArrayOps_static.checkInt(borrowed, sidl_ex);
  init_part(); run_part("borrowed int: checkInt() before copy", resCheckInt1 == true);

  borrowed.smartCopy();
  var resCheckInt2 = ArrayTest.ArrayOps_static.checkInt(borrowed, sidl_ex);
  init_part(); run_part("borrowed int: checkInt() after copy", resCheckInt2 == true);

  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check sidl.borrow_int_Array", sidl_ex);
}

{
  tracker.writeComment("Start: Check sidl.borrow_int_Array:slice", sidl_ex);

  magicNumber = clearstack(magicNumber);
  var borrowed: sidl.Array(int(32), sidl_int__array);
  var elementsExtra: [-1..32] int(32) =
    (-1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
     89, 97, 101, 103, 107, 109, 113, 127, 131, -1);
  var elements = elementsExtra[0..31];

  borrowed = sidl.borrow_int_Array(elements, int_ptr(elements[0]));
  init_part(); run_part("borrowed_int:slice: not-nil", borrowed._not_nil());

  var resCheckInt1 = ArrayTest.ArrayOps_static.checkInt(borrowed, sidl_ex);
  init_part(); run_part("borrowed int:slice: checkInt() before copy", resCheckInt1 == true);

  borrowed.smartCopy();
  var resCheckInt2 = ArrayTest.ArrayOps_static.checkInt(borrowed, sidl_ex);
  init_part(); run_part("borrowed int:slice: checkInt() after copy", resCheckInt2 == true);

  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check sidl.borrow_int_Array:slice", sidl_ex);
}

{
  magicNumber = clearstack(magicNumber);
  var larray: sidl.Array(int(64), sidl_long__array) = ArrayTest.ArrayOps_static.createLong(TEST_SIZE, sidl_ex);
  init_part(); run_part("createLong", larray._not_nil());
  init_part(); run_part("createLong", ArrayTest.ArrayOps_static.checkLong(larray, sidl_ex) == true);
  init_part(); run_part("createLong", ArrayTest.ArrayOps_static.reverseLong(larray, true, sidl_ex) == true);
  larray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var larray: sidl.Array(int(64), sidl_long__array);
  ArrayTest.ArrayOps_static.makeLong(218, larray, sidl_ex);
  init_part(); run_part("makeLong", ArrayTest.ArrayOps_static.checkLong(larray, sidl_ex) == true);
  init_part(); run_part("makeLong", ArrayTest.ArrayOps_static.reverseLong(larray, false, sidl_ex) == true);
  init_part(); run_part("makeLong", ArrayTest.ArrayOps_static.checkLong(larray, sidl_ex) == false);
  larray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var sarray: sidl.Array(string, sidl_string__array);
  sarray = ArrayTest.ArrayOps_static.createString(TEST_SIZE, sidl_ex);
  init_part(); run_part("createString", sarray._not_nil());
  init_part(); run_part("createString", ArrayTest.ArrayOps_static.checkString(sarray, sidl_ex) == true);
  init_part(); run_part("createString", ArrayTest.ArrayOps_static.reverseString(sarray, true, sidl_ex) == true);
  sarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var sarray: sidl.Array(string, sidl_string__array);
  ArrayTest.ArrayOps_static.makeString(218, sarray, sidl_ex);
  init_part(); run_part("makeString", ArrayTest.ArrayOps_static.checkString(sarray, sidl_ex) == true);
  init_part(); run_part("makeString", ArrayTest.ArrayOps_static.reverseString(sarray, false, sidl_ex) == true);
  init_part(); run_part("makeString", ArrayTest.ArrayOps_static.checkString(sarray, sidl_ex) == false);
  sarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var darray: sidl.Array(real(64), sidl_double__array);
  darray = ArrayTest.ArrayOps_static.createDouble(TEST_SIZE, sidl_ex);
  init_part(); run_part("createDouble", darray._not_nil());
  init_part(); run_part("createDouble", ArrayTest.ArrayOps_static.checkDouble(darray, sidl_ex) == true);
  init_part(); run_part("createDouble", ArrayTest.ArrayOps_static.reverseDouble(darray, true, sidl_ex) == true);
  darray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var darray: sidl.Array(real(64), sidl_double__array);
  ArrayTest.ArrayOps_static.makeDouble(218, darray, sidl_ex);
  init_part(); run_part("makeDouble", ArrayTest.ArrayOps_static.checkDouble(darray, sidl_ex) == true);
  init_part(); run_part("makeDouble", ArrayTest.ArrayOps_static.reverseDouble(darray, false, sidl_ex) == true);
  init_part(); run_part("makeDouble", ArrayTest.ArrayOps_static.checkDouble(darray, sidl_ex) == false);
  darray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var farray: sidl.Array(real(32), sidl_float__array);
  farray = ArrayTest.ArrayOps_static.createFloat(TEST_SIZE, sidl_ex);
  init_part(); run_part("createFloat", farray._not_nil());
  init_part(); run_part("createFloat", ArrayTest.ArrayOps_static.checkFloat(farray, sidl_ex) == true);
  init_part(); run_part("createFloat", ArrayTest.ArrayOps_static.reverseFloat(farray, true, sidl_ex) == true);
  farray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var farray: sidl.Array(real(32), sidl_float__array);
  ArrayTest.ArrayOps_static.makeFloat(218, farray, sidl_ex);
  init_part(); run_part("makeFloat", ArrayTest.ArrayOps_static.checkFloat(farray, sidl_ex) == true);
  init_part(); run_part("makeFloat", ArrayTest.ArrayOps_static.reverseFloat(farray, false, sidl_ex) == true);
  init_part(); run_part("makeFloat", ArrayTest.ArrayOps_static.checkFloat(farray, sidl_ex) == false);
  farray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var fcarray: sidl.Array(complex(64), sidl_fcomplex__array);
  fcarray = ArrayTest.ArrayOps_static.createFcomplex(TEST_SIZE, sidl_ex);
  init_part(); run_part("createFcomplex", fcarray._not_nil());
  init_part(); run_part("createFcomplex", ArrayTest.ArrayOps_static.checkFcomplex(fcarray, sidl_ex) == true);
  init_part(); run_part("createFcomplex", ArrayTest.ArrayOps_static.reverseFcomplex(fcarray, true, sidl_ex) == true);
  fcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var fcarray: sidl.Array(complex(64), sidl_fcomplex__array);
  ArrayTest.ArrayOps_static.makeFcomplex(218, fcarray, sidl_ex);
  init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps_static.checkFcomplex(fcarray, sidl_ex) == true);
  init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps_static.reverseFcomplex(fcarray, false, sidl_ex) == true);
  init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps_static.checkFcomplex(fcarray, sidl_ex) == false);
  fcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var dcarray: sidl.Array(complex(128), sidl_dcomplex__array);
  dcarray = ArrayTest.ArrayOps_static.createDcomplex(TEST_SIZE, sidl_ex);
  init_part(); run_part("createDcomplex", dcarray._not_nil());
  init_part(); run_part("createDcomplex", ArrayTest.ArrayOps_static.checkDcomplex(dcarray, sidl_ex) == true);
  init_part(); run_part("createDcomplex", ArrayTest.ArrayOps_static.reverseDcomplex(dcarray, true, sidl_ex) == true);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  magicNumber = clearstack(magicNumber);
  var dcarray: sidl.Array(complex(128), sidl_dcomplex__array);
  ArrayTest.ArrayOps_static.makeDcomplex(218, dcarray, sidl_ex);
  init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps_static.checkDcomplex(dcarray, sidl_ex) == true);
  init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps_static.reverseDcomplex(dcarray, false, sidl_ex) == true);
  init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps_static.checkDcomplex(dcarray, sidl_ex) == false);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create2Int(TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("create2Int", ArrayTest.ArrayOps_static.check2Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
{ 
  magicNumber = clearstack(magicNumber);
  var darray: sidl.Array(real(64), sidl_double__array);
  darray = ArrayTest.ArrayOps_static.create2Double(TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("create2Double", ArrayTest.ArrayOps_static.check2Double(darray, sidl_ex) == true);
  darray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
{ 
  magicNumber = clearstack(magicNumber);
  var farray: sidl.Array(real(32), sidl_float__array);
  farray = ArrayTest.ArrayOps_static.create2Float(TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("create2Float", ArrayTest.ArrayOps_static.check2Float(farray, sidl_ex) == true);
  farray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
{ 
  magicNumber = clearstack(magicNumber);
  var dcarray: sidl.Array(complex(128), sidl_dcomplex__array);
  dcarray = ArrayTest.ArrayOps_static.create2Dcomplex(TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("create2Dcomplex", ArrayTest.ArrayOps_static.check2Dcomplex(dcarray, sidl_ex) == true);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}
{ 
  magicNumber = clearstack(magicNumber);
  var dcarray: sidl.Array(complex(64), sidl_fcomplex__array);
  dcarray = ArrayTest.ArrayOps_static.create2Fcomplex(TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("create2Fcomplex", ArrayTest.ArrayOps_static.check2Fcomplex(dcarray, sidl_ex) == true);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

 
{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create3Int(sidl_ex);
  init_part(); run_part("create3Int", ArrayTest.ArrayOps_static.check3Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
} 

{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create4Int(sidl_ex);
  init_part(); run_part("create4Int", ArrayTest.ArrayOps_static.check4Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create5Int(sidl_ex);
  init_part(); run_part("create5Int", ArrayTest.ArrayOps_static.check5Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create6Int(sidl_ex);
  init_part(); run_part("create6Int", ArrayTest.ArrayOps_static.check6Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{ 
  magicNumber = clearstack(magicNumber);
  var iarray: sidl.Array(int(32), sidl_int__array);
  iarray = ArrayTest.ArrayOps_static.create7Int(sidl_ex);
  init_part(); run_part("create7Int", ArrayTest.ArrayOps_static.check7Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);
}

{
  tracker.writeComment("Start: Check makeInOutBool", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var barray = new sidl.Array(bool, sidl_bool__array, nil);
  ArrayTest.ArrayOps_static.makeInOutBool(barray, 218, sidl_ex);
  init_part(); run_part("makeInOutBool", ArrayTest.ArrayOps_static.checkBool(barray, sidl_ex) == true);
  barray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutBool", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutChar", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var carray = new sidl.Array(string, sidl_char__array, nil);
  ArrayTest.ArrayOps_static.makeInOutChar(carray, 218, sidl_ex);
  init_part(); run_part("makeInOutChar", ArrayTest.ArrayOps_static.checkChar(carray, sidl_ex) == true);
  carray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutChar", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutInt", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOutInt(iarray, 218, sidl_ex);
  init_part(); run_part("makeInOutInt", ArrayTest.ArrayOps_static.checkInt(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutInt", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutLong", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var larray = new sidl.Array(int(64), sidl_long__array, nil);
  ArrayTest.ArrayOps_static.makeInOutLong(larray, 218, sidl_ex);
  init_part(); run_part("makeInOutLong", ArrayTest.ArrayOps_static.checkLong(larray, sidl_ex) == true);
  larray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutLong", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutString", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var sarray = new sidl.Array(string, sidl_string__array, nil);
  ArrayTest.ArrayOps_static.makeInOutString(sarray, 218, sidl_ex);
  init_part(); run_part("makeInOutString", ArrayTest.ArrayOps_static.checkString(sarray, sidl_ex) == true);
  sarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutString", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutDouble", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var darray = new sidl.Array(real(64), sidl_double__array, nil);
  ArrayTest.ArrayOps_static.makeInOutDouble(darray, 218, sidl_ex);
  init_part(); run_part("makeInOutDouble", ArrayTest.ArrayOps_static.checkDouble(darray, sidl_ex) == true);
  darray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutDouble", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutFloat", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var farray = new sidl.Array(real(32), sidl_float__array, nil);
  ArrayTest.ArrayOps_static.makeInOutFloat(farray, 218, sidl_ex);
  init_part(); run_part("makeInOutFloat", ArrayTest.ArrayOps_static.checkFloat(farray, sidl_ex) == true);
  farray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutFloat", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutDcomplex", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var dcarray = new sidl.Array(complex(128), sidl_dcomplex__array, nil);
  ArrayTest.ArrayOps_static.makeInOutDcomplex(dcarray, 218, sidl_ex);
  init_part(); run_part("makeInOutDcomplex", ArrayTest.ArrayOps_static.checkDcomplex(dcarray, sidl_ex) == true);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutDcomplex", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOutFcomplex", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var fcarray = new sidl.Array(complex(64), sidl_fcomplex__array, nil);
  ArrayTest.ArrayOps_static.makeInOutFcomplex(fcarray, 218, sidl_ex);
  init_part(); run_part("makeInOutFcomplex", ArrayTest.ArrayOps_static.checkFcomplex(fcarray, sidl_ex) == true);
  fcarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOutFcomplex", sidl_ex);
}

{
  tracker.writeComment("Start: Check makeInOut2Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut2Int(iarray, TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("makeInOut2Int", ArrayTest.ArrayOps_static.check2Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut2Int", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut2Double", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var darray = new sidl.Array(real(64), sidl_double__array, nil);
  ArrayTest.ArrayOps_static.makeInOut2Double(darray, TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("makeInOut2Double", ArrayTest.ArrayOps_static.check2Double(darray, sidl_ex) == true);
  darray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut2Double", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut2Float", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var farray = new sidl.Array(real(32), sidl_float__array, nil);
  ArrayTest.ArrayOps_static.makeInOut2Float(farray, TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("makeInOut2Float", ArrayTest.ArrayOps_static.check2Float(farray, sidl_ex) == true);
  farray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut2Float", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut2Dcomplex", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var dcarray = new sidl.Array(complex(128), sidl_dcomplex__array, nil);
  ArrayTest.ArrayOps_static.makeInOut2Dcomplex(dcarray, TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("makeInOut2Dcomplex", ArrayTest.ArrayOps_static.check2Dcomplex(dcarray, sidl_ex) == true);
  dcarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut2Dcomplex", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut2Fcomplex", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var fcarray = new sidl.Array(complex(64), sidl_fcomplex__array, nil);
  ArrayTest.ArrayOps_static.makeInOut2Fcomplex(fcarray, TEST_DIM1, TEST_DIM2, sidl_ex);
  init_part(); run_part("makeInOut2Fcomplex", ArrayTest.ArrayOps_static.check2Fcomplex(fcarray, sidl_ex) == true);
  fcarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut2Fcomplex", sidl_ex);
}

{
  tracker.writeComment("Start: Check makeInOut3Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut3Int(iarray, sidl_ex);
  init_part(); run_part("makeInOut3Int", ArrayTest.ArrayOps_static.check3Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut3Int", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut4Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut4Int(iarray, sidl_ex);
  init_part(); run_part("makeInOut4Int", ArrayTest.ArrayOps_static.check4Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut4Int", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut5Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut5Int(iarray, sidl_ex);
  init_part(); run_part("makeInOut5Int", ArrayTest.ArrayOps_static.check5Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut5Int", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut6Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut6Int(iarray, sidl_ex);
  init_part(); run_part("makeInOut6Int", ArrayTest.ArrayOps_static.check6Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut6Int", sidl_ex);
}
{
  tracker.writeComment("Start: Check makeInOut7Int", sidl_ex);

  magicNumber = clearstack(magicNumber);
  // Explicitly instantiate before passing to function
  var iarray = new sidl.Array(int(32), sidl_int__array, nil);
  ArrayTest.ArrayOps_static.makeInOut7Int(iarray, sidl_ex);
  init_part(); run_part("makeInOut7Int", ArrayTest.ArrayOps_static.check7Int(iarray, sidl_ex) == true);
  iarray.deleteRef();
  magicNumber = clearstack(magicNumber);

  tracker.writeComment("End: Check makeInOut7Int", sidl_ex);
}

  /* { */
  /*   const int32_t numElem[] = { (TEST_SIZE+1)/2 }; */
  /*   const int32_t numElemTwo[] = { TEST_SIZE/2 }; */
  /*   const int32_t start[] = { 0 }; */
  /*   const int32_t startTwo[] = { 1 }; */
  /*   const int32_t stride[] = { 2 }; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<ArrayTest::ArrayOps> objarray =  */
  /*     sidl::array<ArrayTest::ArrayOps>::create1d(TEST_SIZE); */
  /*   init_part(); run_part("create1d", objarray._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(objarray, sidl_ex) == 0); */
  /*   for(int32_t i = 0; i < TEST_SIZE; i += 2) { */
  /*     objarray.set(i, ArrayTest.ArrayOps_static._create()); */
  /*   } */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(objarray, sidl_ex) == ((TEST_SIZE+1)/2)); */
  /*   sidl::array<ArrayTest::ArrayOps> sliced =  */
  /*     objarray.slice(1, numElem, start, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(sliced) == ((TEST_SIZE+1)/2)); */
  /*   sliced = objarray.slice(1, numElemTwo, startTwo, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(sliced) == 0); */
  /*   objarray.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(objarray, sidl_ex) == ((TEST_SIZE+1)/2)); */
  /*   sliced.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps_static.checkObject(sliced) == 0); */

  /*   init_part(); run_part("createObjectNegOne", ArrayTest.ArrayOps_static.createObject(-1)._is_nil()); */
  /*   init_part(); run_part("checkObjectNull", !ArrayTest.ArrayOps_static.checkObject(NULL)); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */


  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<bool> ary; */
  /*   init_part(); run_part("createBoolNegOne", ArrayTest.ArrayOps_static.createBool(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeBool(-1, ary); */
  /*   init_part(); run_part("makeBoolNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<char> ary; */
  /*   init_part(); run_part("createCharNegOne", ArrayTest.ArrayOps_static.createChar(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeChar(-1, ary); */
  /*   init_part(); run_part("makeCharNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int32_t> ary; */
  /*   init_part(); run_part("createIntNegOne", ArrayTest.ArrayOps_static.createInt(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeInt(-1, ary); */
  /*   init_part(); run_part("makeIntNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int64_t> ary; */
  /*   init_part(); run_part("createLongNegOne", ArrayTest.ArrayOps_static.createLong(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeLong(-1, ary); */
  /*   init_part(); run_part("makeLongNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<string> ary; */
  /*   init_part(); run_part("createStringNegOne", ArrayTest.ArrayOps_static.createString(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeString(-1, ary); */
  /*   init_part(); run_part("makeStringNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<double> ary; */
  /*   init_part(); run_part("createDoubleNegOne", ArrayTest.ArrayOps_static.createDouble(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeDouble(-1, ary); */
  /*   init_part(); run_part("makeDoubleNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<float> ary; */
  /*   init_part(); run_part("createFloatNegOne", ArrayTest.ArrayOps_static.createFloat(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeFloat(-1, ary); */
  /*   init_part(); run_part("makeFloatNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::dcomplex> ary; */
  /*   init_part(); run_part("createDcomplexNegOne", ArrayTest.ArrayOps_static.createDcomplex(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeDcomplex(-1, ary); */
  /*   init_part(); run_part("makeDcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::fcomplex> ary; */
  /*   init_part(); run_part("createFcomplexNegOne", ArrayTest.ArrayOps_static.createFcomplex(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps_static.makeFcomplex(-1, ary); */
  /*   init_part(); run_part("makeFcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DoubleNegOne", ArrayTest.ArrayOps_static.create2Double(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FcomplexNegOne", ArrayTest.ArrayOps_static.create2Fcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DcomplexNegOne", ArrayTest.ArrayOps_static.create2Dcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FloatNegOne", ArrayTest.ArrayOps_static.create2Float(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2IntNegOne", ArrayTest.ArrayOps_static.create2Int(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   int32_t dimen, dimen2, type, type2 = 0; */
  /*   ::sidl::basearray garray, garrayret, garrayout, garrayinout; */
  /*   ::sidl::array<int32_t> iarray; */
  /*   magicNumber = clearstack(magicNumber); */
    
  /*   ArrayTest.ArrayOps_static.checkGeneric(garray, dimen, type); */
  /*   init_part(); run_part("Generic array is still Null", garray._is_nil()); */
  /*   init_part(); run_part("NULL Generic array has no dimension", (dimen == 0)); */
  /*   init_part(); run_part("NULL Generic array has no type", (type == 0)); */

  /*   dimen = 1; */
  /*   type = sidl_int_array; */
  /*   garray = ArrayTest.ArrayOps_static.createGeneric(dimen, type); */
  /*   init_part(); run_part("Generic (int) array is not Null", garray._not_nil()); */
  /*   init_part(); run_part("Generic (int) array has 1 dimension", (dimen == garray.dimen())); */
  /*   init_part(); run_part("Generic (int) array has int type", type == garray.arrayType()); */
    
  /*   ArrayTest.ArrayOps_static.checkGeneric(garray, dimen2, type2); */
  /*   init_part(); run_part("checkGeneric (int) array has 1 dimension", (dimen == dimen2)); */
  /*   init_part(); run_part("checkGeneric (int) array has int type", (type == type2)); */

  /*   garrayret = ArrayTest.ArrayOps_static.passGeneric(garray, iarray, */
  /* 					       garrayout); */
    
  /*   init_part(); run_part("Generic returned array not NULL", garrayret._not_nil()); */
  /*   init_part(); run_part("Generic returned array correct dimension",  */
  /* 	     dimen == garrayret.dimen()); */
  /*   init_part(); run_part("Generic returned array correct type",  */
  /* 	     type == garrayret.arrayType()); */
  /*   init_part(); run_part("Generic returned array correct length",  */
  /* 	     garray.length(0) == garrayret.length(0)); */

  /*   init_part(); run_part("Generic returned array not NULL", garrayout._not_nil()); */
  /*   init_part(); run_part("Generic returned array correct dimension",  */
  /* 	     dimen == garrayout.dimen()); */
  /*   init_part(); run_part("Generic returned array correct type",  */
  /* 	     type == garrayout.arrayType()); */
  /*   init_part(); run_part("Generic returned array correct length",  */
  /* 	     garray.length(0) == garrayout.length(0)); */

  /*   //iarray = ::sidl::array<int32_t>(garray, sidl_ex); */

  /*   init_part(); run_part("Generic inout is correct",  */
  /* 	     ArrayTest.ArrayOps_static.check2Int(iarray, sidl_ex) == true); */

  /* } */

  {
    tracker.writeComment("Start: Check initialization for borrowed array int 32b - 1D", sidl_ex);

    magicNumber = clearstack(magicNumber);
    var iarray: sidl.Array(int(32), sidl_int__array) = ArrayTest.ArrayOps_static.createInt(TEST_SIZE, sidl_ex);

    init_part(); run_part("createInt-not nil", iarray != nil);
    init_part(); run_part("createInt", ArrayTest.ArrayOps_static.checkInt(iarray, sidl_ex) == true);

    var opData: opaque = iarray.first();
    var ibarray = sidl.createBorrowedArray1d(iarray);

    init_part(); run_part("Check ibarray int 1", ArrayTest.ArrayOps_static.checkRarray1Int(ibarray, TEST_SIZE, sidl_ex) == true);
    
    tracker.writeComment("End: Check initialization for borrowed array int 32b - 1D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for borrowed array int 32b - 3D", sidl_ex);

    magicNumber = clearstack(magicNumber);
    var iarray: sidl.Array(int(32), sidl_int__array) = ArrayTest.ArrayOps_static.create3Int(sidl_ex);

    tracker.writeComment("check not nil", sidl_ex);
    init_part(); run_part("create3Int-not nil", iarray != nil);
    
    tracker.writeComment("check3Int", sidl_ex);
    init_part(); run_part("create3Int", ArrayTest.ArrayOps_static.check3Int(iarray, sidl_ex) == true);

    tracker.writeComment("ensureOrdering", sidl_ex);
    //iarray.ensureOrdering(3, sidl.sidl_array_ordering.sidl_column_major_order);

    tracker.writeComment("createBorrowedArray3d", sidl_ex);
    var opData: opaque = iarray.first();
    var ibarray = sidl.createBorrowedArray3d(iarray);
    var m = iarray.length(0), n = iarray.length(1), o = iarray.length(2);
    
    tracker.writeComment("checkRarray3Int", sidl_ex);
    init_part(); run_part("Check ibarray int 3", 
        ArrayTest.ArrayOps_static.checkRarray3Int(ibarray, m, n, o, sidl_ex) == true);
    
    tracker.writeComment("End: Check initialization for borrowed array int 32b - 3D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for borrowed array int 32b - 7D", sidl_ex);

    magicNumber = clearstack(magicNumber);
    var iarray: sidl.Array(int(32), sidl_int__array) = ArrayTest.ArrayOps_static.create7Int(sidl_ex);

    tracker.writeComment("check not nil", sidl_ex);
    init_part(); run_part("create7Int-not nil", iarray != nil);

    tracker.writeComment("check7Int", sidl_ex);
    init_part(); run_part("create7Int", ArrayTest.ArrayOps_static.check7Int(iarray, sidl_ex) == true);

    tracker.writeComment("ensureOrdering", sidl_ex);
    //iarray.ensureOrdering(7, sidl.sidl_array_ordering.sidl_column_major_order);

    tracker.writeComment("createBorrowedArray7d", sidl_ex);
    var opData: opaque = iarray.first();
    var ibarray = sidl.createBorrowedArray7d(iarray);
    var m = iarray.length(0), n = iarray.length(1), o = iarray.length(2),
      p = iarray.length(3), q = iarray.length(4), r = iarray.length(5), 
      s = iarray.length(6);
    
    init_part(); run_part("Check ibarray int 7", 
        ArrayTest.ArrayOps_static.checkRarray7Int(ibarray, m, n, o, p, q, r, s, sidl_ex) == true);

    tracker.writeComment("End: Check initialization for borrowed array int 32b - 7D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for int 32b - 1D", sidl_ex);

    var irarray: [0.. #TEST_SIZE] int(32);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Int(irarray, TEST_SIZE, sidl_ex);
    init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps_static.checkRarray1Int(irarray, TEST_SIZE, sidl_ex) == true);
    tracker.writeComment("End: Check initialization for int 32b - 1D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for int 32b - 3D", sidl_ex);

    var n = 2, m = 3, o = 4;    
    var irarray: [0.. #n, 0.. #m, 0.. #o] int(32); //2*3*4 

    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray3Int(irarray, n, m, o, sidl_ex);
    init_part(); run_part("Check rarray int 2", ArrayTest.ArrayOps_static.checkRarray3Int(irarray, n, m, o, sidl_ex) == true);

    tracker.writeComment("End: Check initialization for int 32b - 3D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for int 32b - 7D", sidl_ex);

    var n = 2, m = 2 , o = 2, p = 2, q = 3, r = 3, s = 3;
    var irarray: [0.. #n, 0.. #m, 0.. #o, 0.. #p, 0.. #q, 0.. #r, 0.. #s] int(32); //2*2*2*2*3*3*3 

    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray7Int(irarray, n, m, o, p, q, r, s, sidl_ex);
    init_part(); run_part("Check rarray int 7", ArrayTest.ArrayOps_static.checkRarray7Int(irarray, n, m, o, p, q, r, s, sidl_ex) == true);

    tracker.writeComment("End: Check initialization for int 32b - 7D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for real 64b - 1D", sidl_ex);

    var irarray: [0.. #TEST_SIZE] real(64);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Double(irarray, TEST_SIZE, sidl_ex);
    init_part(); run_part("Check rarray double 1", ArrayTest.ArrayOps_static.checkRarray1Double(irarray, TEST_SIZE, sidl_ex) == true);

    tracker.writeComment("End: Check initialization for real 64b - 1D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check initialization for complex with 64b components - 1D", sidl_ex);

    var irarray: [0.. #TEST_SIZE] complex(128);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Dcomplex(irarray, TEST_SIZE, sidl_ex);
    init_part(); run_part("Check rarray dcomplex 1", ArrayTest.ArrayOps_static.checkRarray1Dcomplex(irarray, TEST_SIZE, sidl_ex) == true);

    tracker.writeComment("End: Check initialization for complex with 64b components - 1D", sidl_ex);
  }

  {
    tracker.writeComment("Start: Check matrix multiplication", sidl_ex);
   
    var n = 3, m = 3, o = 2;
    var a: [0.. #n, 0.. #m] int(32);
    var b: [0.. #m, 0.. #o] int(32);
    var x: [0.. #n, 0.. #o] int(32);

    [(i) in [0..8]] a[i / m, i % m] = i;
    [(i) in [0..5]] b[i / o, i % o] = i;

    tracker.writeComment("matrixMultiply()", sidl_ex);
    ArrayTest.ArrayOps_static.matrixMultiply(a, b, x, n, m, o, sidl_ex);

    tracker.writeComment("checkMatrixMultiply()", sidl_ex);
    init_part(); run_part("Check Matrix Multiply", ArrayTest.ArrayOps_static.checkMatrixMultiply(a, b, x, n, m, o, sidl_ex) == true);

    tracker.writeComment("End: Check matrix multiplication", sidl_ex);
 }

 {
    tracker.writeComment("Start: Check matrix multiplication:slice", sidl_ex);

    var n = 3, m = 3, o = 2;
    var ae: [-2..n, -2..m] int(32);
    var be: [-2..m, 0..o] int(32);
    var xe: [-2..n, 0..o] int(32);

    [(i) in [0..8]] ae[i / m, i % m] = i;
    [(i) in [0..5]] be[i / o, i % o] = i;

    tracker.writeComment("matrixMultiply():slice", sidl_ex);
    ArrayTest.ArrayOps_static.matrixMultiply(
        ae[0.. #n, 0.. #m],
        be[0.. #m, 0.. #o],
        xe[0.. #n, 0.. #o],
        n, m, o, sidl_ex);

    tracker.writeComment("checkMatrixMultiply():slice", sidl_ex);
    init_part(); run_part("Check Matrix Multiply", ArrayTest.ArrayOps_static.checkMatrixMultiply(
        ae[0.. #n, 0.. #m],
        be[0.. #m, 0.. #o],
        xe[0.. #n, 0.. #o],
        n, m, o, sidl_ex) == true);

    tracker.writeComment("End: Check matrix multiplication:slice", sidl_ex);
 }

  
 {
   tracker.writeComment("Start: Check create2String", sidl_ex);

   magicNumber = clearstack(magicNumber);
   var sarray: sidl.Array(string, sidl_string__array) = ArrayTest.ArrayOps_static.create2String(12,13, sidl_ex);
   init_part(); run_part("createString", sarray._not_nil());
   init_part(); run_part("createString", ArrayTest.ArrayOps_static.check2String(sarray, sidl_ex) == true);
   sarray.deleteRef();
   magicNumber = clearstack(magicNumber);

   tracker.writeComment("End: Check create2String", sidl_ex);
 }


tracker.close(sidl_ex);
