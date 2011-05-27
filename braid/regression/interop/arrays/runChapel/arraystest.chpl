// 
// File:        arraytests.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Simple test on the ArrayTest static methods
// 
use ArrayTest;
use synch;
use sidl;

var part_no: int = 0;
var tracker: synch.RegOut = synch.RegOut_static.getInstance();
var magic_number = 13;

proc init_part()
{
  part_no += 1;
  tracker.startPart(part_no);
  tracker.writeComment("Part "+part_no);
}

proc run_part(msg: string, result: bool)
{
  var r: ResultType;
  tracker.writeComment(msg);
  if (result) then
    r = ResultType.PASS;
  else 
    r = ResultType.FAIL;
  tracker.endPart(part_no, r);
  tracker.writeComment("End of part "+part_no);
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
tracker.setExpectations(-1);
  
//tracker.setExpectations(158);

  /* { */
  /*   const int32_t numElem[] = { TEST_SIZE/2 }; */
  /*   const int32_t start[] = { 0 }; */
  /*   const int32_t stride[] = { 2 }; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray = ArrayTest.ArrayOps.createBool(TEST_SIZE); */
  /*   init_part(); run_part("createBool", barray._not_nil()); */
  /*   init_part(); run_part("createBool", ArrayTest.ArrayOps.checkBool(barray) == true); */
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

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray; */
  /*   ArrayTest.ArrayOps.makeBool(218, barray); */
  /*   init_part(); run_part("makeBool218", ArrayTest.ArrayOps.checkBool(barray) == true); */
  /*   init_part(); run_part("makeBool218", ArrayTest.ArrayOps.reverseBool(barray, false) == true); */
  /*   init_part(); run_part("makeBool218", ArrayTest.ArrayOps.checkBool(barray) == false); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray; */
  /*   ArrayTest.ArrayOps.makeBool(9, barray); */
  /*   init_part(); run_part("makeBool9", ArrayTest.ArrayOps.reverseBool(barray, false) == true); */
  /*   init_part(); run_part("makeBool9", ArrayTest.ArrayOps.checkBool(barray) == true); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray = ArrayTest.ArrayOps.createChar(TEST_SIZE); */
  /*   init_part(); run_part("createChar", carray._not_nil()); */
  /*   init_part(); run_part("createChar", ArrayTest.ArrayOps.checkChar(carray) == true); */
  /*   init_part(); run_part("createChar", ArrayTest.ArrayOps.reverseChar(carray, true) == true); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray; */
  /*   ArrayTest.ArrayOps.makeChar(218, carray); */
  /*   init_part(); run_part("makeChar", ArrayTest.ArrayOps.checkChar(carray) == true); */
  /*   init_part(); run_part("makeChar", ArrayTest.ArrayOps.reverseChar(carray, false) == true); */
  /*   init_part(); run_part("makeChar", ArrayTest.ArrayOps.checkChar(carray) == false); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayTest.ArrayOps.createInt(TEST_SIZE); */
  /*   init_part(); run_part("createInt", iarray._not_nil()); */
  /*   init_part(); run_part("createInt", ArrayTest.ArrayOps.checkInt(iarray) == true); */
  /*   init_part(); run_part("createInt", ArrayTest.ArrayOps.reverseInt(iarray, true) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

{
   magicNumber = clearstack(magicNumber);
   var iarray: sidl.Array(int(32));
   ArrayTest.ArrayOps_static.makeInt(218, iarray);
   init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.checkInt(iarray) == true);
   init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.reverseInt(iarray, false) == true);
   init_part(); run_part("makeInt", ArrayTest.ArrayOps_static.checkInt(iarray) == false);
   iarray.deleteRef();
   magicNumber = clearstack(magicNumber);
}

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> borrowed; */
  /*   int32_t elements[] = { */
  /*     2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, */
  /*     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, */
  /*     89, 97, 101, 103, 107, 109, 113, 127, 131 */
  /*   }; */
  /*   const int32_t lower[] = { 0 }; */
  /*   const int32_t upper[] = { sizeof(elements)/sizeof(int32_t)-1 }; */
  /*   const int32_t stride[] = { 1 }; */
  /*   init_part(); run_part("borrowed_int", !borrowed); */
  /*   borrowed.borrow(elements, 1, lower, upper, stride); */
  /*   init_part(); run_part("borrowed_int", borrowed._not_nil()); */
  /*   init_part(); run_part("borrowed int", ArrayTest.ArrayOps.checkInt(borrowed) == true); */
  /*   borrowed.smartCopy(); */
  /*   init_part(); run_part("borrowed int", ArrayTest.ArrayOps.checkInt(borrowed) == true); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray = ArrayTest.ArrayOps.createLong(TEST_SIZE); */
  /*   init_part(); run_part("createLong", larray._not_nil()); */
  /*   init_part(); run_part("createLong", ArrayTest.ArrayOps.checkLong(larray) == true); */
  /*   init_part(); run_part("createLong", ArrayTest.ArrayOps.reverseLong(larray, true) == true); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray; */
  /*   ArrayTest.ArrayOps.makeLong(218, larray); */
  /*   init_part(); run_part("makeLong", ArrayTest.ArrayOps.checkLong(larray) == true); */
  /*   init_part(); run_part("makeLong", ArrayTest.ArrayOps.reverseLong(larray, false) == true); */
  /*   init_part(); run_part("makeLong", ArrayTest.ArrayOps.checkLong(larray) == false); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray = ArrayTest.ArrayOps.createString(TEST_SIZE); */
  /*   init_part(); run_part("createString", sarray._not_nil()); */
  /*   init_part(); run_part("createString", ArrayTest.ArrayOps.checkString(sarray) == true); */
  /*   init_part(); run_part("createString", ArrayTest.ArrayOps.reverseString(sarray, true) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray;  */
  /*   ArrayTest.ArrayOps.makeString(218, sarray); */
  /*   init_part(); run_part("makeString", ArrayTest.ArrayOps.checkString(sarray) == true); */
  /*   init_part(); run_part("makeString", ArrayTest.ArrayOps.reverseString(sarray, false) == true); */
  /*   init_part(); run_part("makeString", ArrayTest.ArrayOps.checkString(sarray) == false); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray = ArrayTest.ArrayOps.createDouble(TEST_SIZE); */
  /*   init_part(); run_part("createDouble", darray._not_nil()); */
  /*   init_part(); run_part("createDouble", ArrayTest.ArrayOps.checkDouble(darray) == true); */
  /*   init_part(); run_part("createDouble", ArrayTest.ArrayOps.reverseDouble(darray, true) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayTest.ArrayOps.makeDouble(218, darray); */
  /*   init_part(); run_part("makeDouble", ArrayTest.ArrayOps.checkDouble(darray) == true); */
  /*   init_part(); run_part("makeDouble", ArrayTest.ArrayOps.reverseDouble(darray, false) == true); */
  /*   init_part(); run_part("makeDouble", ArrayTest.ArrayOps.checkDouble(darray) == false); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray = ArrayTest.ArrayOps.createFloat(TEST_SIZE); */
  /*   init_part(); run_part("createFloat", farray._not_nil()); */
  /*   init_part(); run_part("createFloat", ArrayTest.ArrayOps.checkFloat(farray) == true); */
  /*   init_part(); run_part("createFloat", ArrayTest.ArrayOps.reverseFloat(farray, true) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray;  */
  /*   ArrayTest.ArrayOps.makeFloat(218, farray); */
  /*   init_part(); run_part("makeFloat", ArrayTest.ArrayOps.checkFloat(farray) == true); */
  /*   init_part(); run_part("makeFloat", ArrayTest.ArrayOps.reverseFloat(farray, false) == true); */
  /*   init_part(); run_part("makeFloat", ArrayTest.ArrayOps.checkFloat(farray) == false); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray = ArrayTest.ArrayOps.createFcomplex(TEST_SIZE); */
  /*   init_part(); run_part("createFcomplex", fcarray._not_nil()); */
  /*   init_part(); run_part("createFcomplex", ArrayTest.ArrayOps.checkFcomplex(fcarray) == true); */
  /*   init_part(); run_part("createFcomplex", ArrayTest.ArrayOps.reverseFcomplex(fcarray, true) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayTest.ArrayOps.makeFcomplex(218, fcarray); */
  /*   init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps.checkFcomplex(fcarray) == true); */
  /*   init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps.reverseFcomplex(fcarray, false) == true); */
  /*   init_part(); run_part("makeFcomplex", ArrayTest.ArrayOps.checkFcomplex(fcarray) == false); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray = ArrayTest.ArrayOps.createDcomplex(TEST_SIZE); */
  /*   init_part(); run_part("createDcomplex", dcarray._not_nil()); */
  /*   init_part(); run_part("createDcomplex", ArrayTest.ArrayOps.checkDcomplex(dcarray) == true); */
  /*   init_part(); run_part("createDcomplex", ArrayTest.ArrayOps.reverseDcomplex(dcarray, true) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayTest.ArrayOps.makeDcomplex(218, dcarray); */
  /*   init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps.checkDcomplex(dcarray) == true); */
  /*   init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps.reverseDcomplex(dcarray, false) == true); */
  /*   init_part(); run_part("makeDcomplex", ArrayTest.ArrayOps.checkDcomplex(dcarray) == false); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   iarray = ArrayTest.ArrayOps.create2Int(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Int", ArrayTest.ArrayOps.check2Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   darray = ArrayTest.ArrayOps.create2Double(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Double", ArrayTest.ArrayOps.check2Double(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   farray = ArrayTest.ArrayOps.create2Float(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Float", ArrayTest.ArrayOps.check2Float(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   dcarray = ArrayTest.ArrayOps.create2Dcomplex(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Dcomplex", ArrayTest.ArrayOps.check2Dcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   fcarray = ArrayTest.ArrayOps.create2Fcomplex(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Fcomplex", ArrayTest.ArrayOps.check2Fcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayTest.ArrayOps.create3Int(); */
  /*   init_part(); run_part("create3Int", ArrayTest.ArrayOps.check3Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayTest.ArrayOps.create4Int(); */
  /*   init_part(); run_part("create4Int", ArrayTest.ArrayOps.check4Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayTest.ArrayOps.create5Int(); */
  /*   init_part(); run_part("create5Int", ArrayTest.ArrayOps.check5Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayTest.ArrayOps.create6Int(); */
  /*   init_part(); run_part("create6Int", ArrayTest.ArrayOps.check6Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayTest.ArrayOps.create7Int(); */
  /*   init_part(); run_part("create7Int", ArrayTest.ArrayOps.check7Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray; */
  /*   ArrayTest.ArrayOps.makeInOutBool(barray, 218); */
  /*   init_part(); run_part("makeInOutBool", ArrayTest.ArrayOps.checkBool(barray) == true); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray; */
  /*   ArrayTest.ArrayOps.makeInOutChar(carray, 218); */
  /*   init_part(); run_part("makeInOutChar", ArrayTest.ArrayOps.checkChar(carray) == true); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   ArrayTest.ArrayOps.makeInOutInt(iarray, 218); */
  /*   init_part(); run_part("makeInOutInt", ArrayTest.ArrayOps.checkInt(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray; */
  /*   ArrayTest.ArrayOps.makeInOutLong(larray, 218); */
  /*   init_part(); run_part("makeInOutLong", ArrayTest.ArrayOps.checkLong(larray) == true); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray; */
  /*   ArrayTest.ArrayOps.makeInOutString(sarray, 218); */
  /*   init_part(); run_part("makeInOutString", ArrayTest.ArrayOps.checkString(sarray) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayTest.ArrayOps.makeInOutDouble(darray, 218); */
  /*   init_part(); run_part("makeInOutDouble", ArrayTest.ArrayOps.checkDouble(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   ArrayTest.ArrayOps.makeInOutFloat(farray, 218); */
  /*   init_part(); run_part("makeInOutFloat", ArrayTest.ArrayOps.checkFloat(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayTest.ArrayOps.makeInOutDcomplex(dcarray, 218); */
  /*   init_part(); run_part("makeInOutDcomplex", ArrayTest.ArrayOps.checkDcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayTest.ArrayOps.makeInOutFcomplex(fcarray, 218); */
  /*   init_part(); run_part("makeInOutFcomplex", ArrayTest.ArrayOps.checkFcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut2Int(iarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Int", ArrayTest.ArrayOps.check2Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayTest.ArrayOps.makeInOut2Double(darray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Double", ArrayTest.ArrayOps.check2Double(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   ArrayTest.ArrayOps.makeInOut2Float(farray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Float", ArrayTest.ArrayOps.check2Float(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayTest.ArrayOps.makeInOut2Dcomplex(dcarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Dcomplex", ArrayTest.ArrayOps.check2Dcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayTest.ArrayOps.makeInOut2Fcomplex(fcarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Fcomplex", ArrayTest.ArrayOps.check2Fcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut3Int(iarray); */
  /*   init_part(); run_part("makeInOut3Int", ArrayTest.ArrayOps.check3Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut4Int(iarray); */
  /*   init_part(); run_part("makeInOut4Int", ArrayTest.ArrayOps.check4Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut5Int(iarray); */
  /*   init_part(); run_part("makeInOut5Int", ArrayTest.ArrayOps.check5Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut6Int(iarray); */
  /*   init_part(); run_part("makeInOut6Int", ArrayTest.ArrayOps.check6Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayTest.ArrayOps.makeInOut7Int(iarray); */
  /*   init_part(); run_part("makeInOut7Int", ArrayTest.ArrayOps.check7Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

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
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(objarray) == 0); */
  /*   for(int32_t i = 0; i < TEST_SIZE; i += 2) { */
  /*     objarray.set(i, ArrayTest.ArrayOps._create()); */
  /*   } */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(objarray) == ((TEST_SIZE+1)/2)); */
  /*   sidl::array<ArrayTest::ArrayOps> sliced =  */
  /*     objarray.slice(1, numElem, start, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(sliced) == ((TEST_SIZE+1)/2)); */
  /*   sliced = objarray.slice(1, numElemTwo, startTwo, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(sliced) == 0); */
  /*   objarray.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(objarray) == ((TEST_SIZE+1)/2)); */
  /*   sliced.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayTest.ArrayOps.checkObject(sliced) == 0); */

  /*   init_part(); run_part("createObjectNegOne", ArrayTest.ArrayOps.createObject(-1)._is_nil()); */
  /*   init_part(); run_part("checkObjectNull", !ArrayTest.ArrayOps.checkObject(NULL)); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */


  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<bool> ary; */
  /*   init_part(); run_part("createBoolNegOne", ArrayTest.ArrayOps.createBool(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeBool(-1, ary); */
  /*   init_part(); run_part("makeBoolNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<char> ary; */
  /*   init_part(); run_part("createCharNegOne", ArrayTest.ArrayOps.createChar(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeChar(-1, ary); */
  /*   init_part(); run_part("makeCharNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int32_t> ary; */
  /*   init_part(); run_part("createIntNegOne", ArrayTest.ArrayOps.createInt(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeInt(-1, ary); */
  /*   init_part(); run_part("makeIntNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int64_t> ary; */
  /*   init_part(); run_part("createLongNegOne", ArrayTest.ArrayOps.createLong(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeLong(-1, ary); */
  /*   init_part(); run_part("makeLongNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<string> ary; */
  /*   init_part(); run_part("createStringNegOne", ArrayTest.ArrayOps.createString(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeString(-1, ary); */
  /*   init_part(); run_part("makeStringNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<double> ary; */
  /*   init_part(); run_part("createDoubleNegOne", ArrayTest.ArrayOps.createDouble(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeDouble(-1, ary); */
  /*   init_part(); run_part("makeDoubleNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<float> ary; */
  /*   init_part(); run_part("createFloatNegOne", ArrayTest.ArrayOps.createFloat(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeFloat(-1, ary); */
  /*   init_part(); run_part("makeFloatNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::dcomplex> ary; */
  /*   init_part(); run_part("createDcomplexNegOne", ArrayTest.ArrayOps.createDcomplex(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeDcomplex(-1, ary); */
  /*   init_part(); run_part("makeDcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::fcomplex> ary; */
  /*   init_part(); run_part("createFcomplexNegOne", ArrayTest.ArrayOps.createFcomplex(-1)._is_nil()); */
  /*   ArrayTest.ArrayOps.makeFcomplex(-1, ary); */
  /*   init_part(); run_part("makeFcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DoubleNegOne", ArrayTest.ArrayOps.create2Double(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FcomplexNegOne", ArrayTest.ArrayOps.create2Fcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DcomplexNegOne", ArrayTest.ArrayOps.create2Dcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FloatNegOne", ArrayTest.ArrayOps.create2Float(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2IntNegOne", ArrayTest.ArrayOps.create2Int(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   int32_t dimen, dimen2, type, type2 = 0; */
  /*   ::sidl::basearray garray, garrayret, garrayout, garrayinout; */
  /*   ::sidl::array<int32_t> iarray; */
  /*   magicNumber = clearstack(magicNumber); */
    
  /*   ArrayTest.ArrayOps.checkGeneric(garray, dimen, type); */
  /*   init_part(); run_part("Generic array is still Null", garray._is_nil()); */
  /*   init_part(); run_part("NULL Generic array has no dimension", (dimen == 0)); */
  /*   init_part(); run_part("NULL Generic array has no type", (type == 0)); */

  /*   dimen = 1; */
  /*   type = sidl_int_array; */
  /*   garray = ArrayTest.ArrayOps.createGeneric(dimen, type); */
  /*   init_part(); run_part("Generic (int) array is not Null", garray._not_nil()); */
  /*   init_part(); run_part("Generic (int) array has 1 dimension", (dimen == garray.dimen())); */
  /*   init_part(); run_part("Generic (int) array has int type", type == garray.arrayType()); */
    
  /*   ArrayTest.ArrayOps.checkGeneric(garray, dimen2, type2); */
  /*   init_part(); run_part("checkGeneric (int) array has 1 dimension", (dimen == dimen2)); */
  /*   init_part(); run_part("checkGeneric (int) array has int type", (type == type2)); */

  /*   garrayret = ArrayTest.ArrayOps.passGeneric(garray, iarray, */
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

  /*   //iarray = ::sidl::array<int32_t>(garray); */

  /*   init_part(); run_part("Generic inout is correct",  */
  /* 	     ArrayTest.ArrayOps.check2Int(iarray) == true); */

  /* } */
  /* { */
  /*   int32_t* irarray = NULL; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayTest.ArrayOps.createInt(TEST_SIZE); */

  /*   init_part(); run_part("createInt", iarray); */
  /*   init_part(); run_part("createInt", ArrayTest.ArrayOps.checkInt(iarray) == true); */

  /*   irarray = iarray.first();//->d_firstElement; */
  /*   init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps.checkRarray1Int(irarray, TEST_SIZE) == true); */
    
  /* } */

  /* { */
  /*   int32_t* irarray = NULL; */
  /*   int32_t n,m,o; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayTest.ArrayOps.create3Int();//iarray = ArrayTest.ArrayOps.create3Int(); */
  /*   init_part(); run_part("create3Int", ArrayTest.ArrayOps.check3Int(iarray) == true); */
  /*   iarray.ensure(3,sidl::column_major_order); */
  /*   irarray = iarray.first();//->d_firstElement; */
  /*   n = iarray.length(0); */
  /*   m = iarray.length(1); */
  /*   o = iarray.length(2); */
  /*   init_part(); run_part("Check rarray int 3", ArrayTest.ArrayOps.checkRarray3Int(irarray, n,m,o) == true); */
 
  /* } */

  /*   { */
  /*   int32_t* irarray = NULL; */
  /*   int32_t n,m,o,p,q,r,s; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayTest.ArrayOps.create7Int();//iarray = ArrayTest.ArrayOps.create7Int(); */
  /*   init_part(); run_part("create3Int", ArrayTest.ArrayOps.check7Int(iarray) == true); */
  /*   iarray.ensure(7,sidl::column_major_order); */
  /*   irarray = iarray.first();//->d_firstElement; */
  /*   n = iarray.length(0); */
  /*   m = iarray.length(1); */
  /*   o = iarray.length(2); */
  /*   p = iarray.length(3); */
  /*   q = iarray.length(4); */
  /*   r = iarray.length(5); */
  /*   s = iarray.length(6); */
  /*   init_part(); run_part("Check rarray int 7", ArrayTest.ArrayOps.checkRarray7Int(irarray, n,m,o,p,q,r,s) == true); */

  /* } */


  {
    tracker.writeComment("Start: Check initialization for int 32b - 1D");

    var irarray: [0..TEST_SIZE - 1] int(32);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Int(irarray, TEST_SIZE);
    init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps_static.checkRarray1Int(irarray, TEST_SIZE) == true);

    tracker.writeComment("End: Check initialization for int 32b - 1D");
  }

  {
    tracker.writeComment("Start: Check initialization for int 32b - 3D");

    var n = 2, m = 3, o = 4;    
    var irarray: [0.. #n, 0.. #m, 0.. #o] int(32); //2*3*4 

    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray3Int(irarray, n, m, o);
    init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps_static.checkRarray3Int(irarray, n, m, o) == true);

    tracker.writeComment("End: Check initialization for int 32b - 3D");
  }

  {
    tracker.writeComment("Start: Check initialization for int 32b - 7D");

    var n = 2, m = 2 , o = 2, p = 2, q = 3, r = 3, s = 3;
    var irarray: [0.. #n, 0.. #m, 0.. #o, 0.. #p, 0.. #q, 0.. #r, 0.. #s] int(32); //2*2*2*2*3*3*3 

    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray7Int(irarray, n, m, o, p, q, r, s);
    init_part(); run_part("Check rarray int 7", ArrayTest.ArrayOps_static.checkRarray7Int(irarray, n, m, o, p, q, r, s) == true);

    tracker.writeComment("End: Check initialization for int 32b - 7D");
  }

  {
    tracker.writeComment("Start: Check initialization for real 64b - 1D");

    var irarray: [0..TEST_SIZE - 1] real(64);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Double(irarray, TEST_SIZE);
    init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps_static.checkRarray1Double(irarray, TEST_SIZE) == true);

    tracker.writeComment("End: Check initialization for real 64b - 1D");
  }

  {
    tracker.writeComment("Start: Check initialization for complex with 64b components - 1D");

    var irarray: [0..TEST_SIZE - 1] complex(128);
    magicNumber = clearstack(magicNumber);
    ArrayTest.ArrayOps_static.initRarray1Dcomplex(irarray, TEST_SIZE);
    init_part(); run_part("Check rarray int 1", ArrayTest.ArrayOps_static.checkRarray1Dcomplex(irarray, TEST_SIZE) == true);

    tracker.writeComment("End: Check initialization for complex with 64b components - 1D");
  }
/*
  {
    tracker.writeComment("Start: Check matrix multiplication");
   
    var n = 3, m = 3, o = 2;
    var a: [0.. #9] int(32);
    var b: [0.. #6] int(32);
    var x: [0.. #6] int(32);

    [(i) in [0..8]] a[i] = i;
    [(i) in [0..5]] b[i] = i;

    ArrayTest.ArrayOps_static.matrixMultiply(a, b, x, n, m, o);
    init_part(); run_part("Check Matrix Multiply", ArrayTest.ArrayOps_static.checkMatrixMultiply(a, b, x, n, m, o) == true);

    tracker.writeComment("End: Check matrix multiplication");
 }
*/
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray = ArrayTest.ArrayOps.create2String(12,13); */
  /*   init_part(); run_part("createString", sarray._not_nil()); */
  /*   init_part(); run_part("createString", ArrayTest.ArrayOps.check2String(sarray) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */


tracker.close();
