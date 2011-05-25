// 
// File:        arraytests.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Simple test on the ArrayTest static methods
// 
use ArrayTest;
use synch;

var part_no: int = 0;
var tracker: synch.RegOut = synch.getInstance();

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
  /*   sidl::array<bool> barray = ArrayOps::createBool(TEST_SIZE); */
  /*   init_part(); run_part("createBool", barray._not_nil()); */
  /*   init_part(); run_part("createBool", ArrayOps::checkBool(barray) == true); */
  /*   init_part(); run_part("createBool", ArrayOps::reverseBool(barray, true) == true); */
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
  /*   ArrayOps::makeBool(218, barray); */
  /*   init_part(); run_part("makeBool218", ArrayOps::checkBool(barray) == true); */
  /*   init_part(); run_part("makeBool218", ArrayOps::reverseBool(barray, false) == true); */
  /*   init_part(); run_part("makeBool218", ArrayOps::checkBool(barray) == false); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray; */
  /*   ArrayOps::makeBool(9, barray); */
  /*   init_part(); run_part("makeBool9", ArrayOps::reverseBool(barray, false) == true); */
  /*   init_part(); run_part("makeBool9", ArrayOps::checkBool(barray) == true); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray = ArrayOps::createChar(TEST_SIZE); */
  /*   init_part(); run_part("createChar", carray._not_nil()); */
  /*   init_part(); run_part("createChar", ArrayOps::checkChar(carray) == true); */
  /*   init_part(); run_part("createChar", ArrayOps::reverseChar(carray, true) == true); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray; */
  /*   ArrayOps::makeChar(218, carray); */
  /*   init_part(); run_part("makeChar", ArrayOps::checkChar(carray) == true); */
  /*   init_part(); run_part("makeChar", ArrayOps::reverseChar(carray, false) == true); */
  /*   init_part(); run_part("makeChar", ArrayOps::checkChar(carray) == false); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayOps::createInt(TEST_SIZE); */
  /*   init_part(); run_part("createInt", iarray._not_nil()); */
  /*   init_part(); run_part("createInt", ArrayOps::checkInt(iarray) == true); */
  /*   init_part(); run_part("createInt", ArrayOps::reverseInt(iarray, true) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   ArrayOps::makeInt(218, iarray); */
  /*   init_part(); run_part("makeInt", ArrayOps::checkInt(iarray) == true); */
  /*   init_part(); run_part("makeInt", ArrayOps::reverseInt(iarray, false) == true); */
  /*   init_part(); run_part("makeInt", ArrayOps::checkInt(iarray) == false); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

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
  /*   init_part(); run_part("borrowed int", ArrayOps::checkInt(borrowed) == true); */
  /*   borrowed.smartCopy(); */
  /*   init_part(); run_part("borrowed int", ArrayOps::checkInt(borrowed) == true); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray = ArrayOps::createLong(TEST_SIZE); */
  /*   init_part(); run_part("createLong", larray._not_nil()); */
  /*   init_part(); run_part("createLong", ArrayOps::checkLong(larray) == true); */
  /*   init_part(); run_part("createLong", ArrayOps::reverseLong(larray, true) == true); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray; */
  /*   ArrayOps::makeLong(218, larray); */
  /*   init_part(); run_part("makeLong", ArrayOps::checkLong(larray) == true); */
  /*   init_part(); run_part("makeLong", ArrayOps::reverseLong(larray, false) == true); */
  /*   init_part(); run_part("makeLong", ArrayOps::checkLong(larray) == false); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray = ArrayOps::createString(TEST_SIZE); */
  /*   init_part(); run_part("createString", sarray._not_nil()); */
  /*   init_part(); run_part("createString", ArrayOps::checkString(sarray) == true); */
  /*   init_part(); run_part("createString", ArrayOps::reverseString(sarray, true) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray;  */
  /*   ArrayOps::makeString(218, sarray); */
  /*   init_part(); run_part("makeString", ArrayOps::checkString(sarray) == true); */
  /*   init_part(); run_part("makeString", ArrayOps::reverseString(sarray, false) == true); */
  /*   init_part(); run_part("makeString", ArrayOps::checkString(sarray) == false); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray = ArrayOps::createDouble(TEST_SIZE); */
  /*   init_part(); run_part("createDouble", darray._not_nil()); */
  /*   init_part(); run_part("createDouble", ArrayOps::checkDouble(darray) == true); */
  /*   init_part(); run_part("createDouble", ArrayOps::reverseDouble(darray, true) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayOps::makeDouble(218, darray); */
  /*   init_part(); run_part("makeDouble", ArrayOps::checkDouble(darray) == true); */
  /*   init_part(); run_part("makeDouble", ArrayOps::reverseDouble(darray, false) == true); */
  /*   init_part(); run_part("makeDouble", ArrayOps::checkDouble(darray) == false); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray = ArrayOps::createFloat(TEST_SIZE); */
  /*   init_part(); run_part("createFloat", farray._not_nil()); */
  /*   init_part(); run_part("createFloat", ArrayOps::checkFloat(farray) == true); */
  /*   init_part(); run_part("createFloat", ArrayOps::reverseFloat(farray, true) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray;  */
  /*   ArrayOps::makeFloat(218, farray); */
  /*   init_part(); run_part("makeFloat", ArrayOps::checkFloat(farray) == true); */
  /*   init_part(); run_part("makeFloat", ArrayOps::reverseFloat(farray, false) == true); */
  /*   init_part(); run_part("makeFloat", ArrayOps::checkFloat(farray) == false); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray = ArrayOps::createFcomplex(TEST_SIZE); */
  /*   init_part(); run_part("createFcomplex", fcarray._not_nil()); */
  /*   init_part(); run_part("createFcomplex", ArrayOps::checkFcomplex(fcarray) == true); */
  /*   init_part(); run_part("createFcomplex", ArrayOps::reverseFcomplex(fcarray, true) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayOps::makeFcomplex(218, fcarray); */
  /*   init_part(); run_part("makeFcomplex", ArrayOps::checkFcomplex(fcarray) == true); */
  /*   init_part(); run_part("makeFcomplex", ArrayOps::reverseFcomplex(fcarray, false) == true); */
  /*   init_part(); run_part("makeFcomplex", ArrayOps::checkFcomplex(fcarray) == false); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray = ArrayOps::createDcomplex(TEST_SIZE); */
  /*   init_part(); run_part("createDcomplex", dcarray._not_nil()); */
  /*   init_part(); run_part("createDcomplex", ArrayOps::checkDcomplex(dcarray) == true); */
  /*   init_part(); run_part("createDcomplex", ArrayOps::reverseDcomplex(dcarray, true) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayOps::makeDcomplex(218, dcarray); */
  /*   init_part(); run_part("makeDcomplex", ArrayOps::checkDcomplex(dcarray) == true); */
  /*   init_part(); run_part("makeDcomplex", ArrayOps::reverseDcomplex(dcarray, false) == true); */
  /*   init_part(); run_part("makeDcomplex", ArrayOps::checkDcomplex(dcarray) == false); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   iarray = ArrayOps::create2Int(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Int", ArrayOps::check2Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   darray = ArrayOps::create2Double(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Double", ArrayOps::check2Double(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   farray = ArrayOps::create2Float(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Float", ArrayOps::check2Float(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   dcarray = ArrayOps::create2Dcomplex(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Dcomplex", ArrayOps::check2Dcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* {  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   fcarray = ArrayOps::create2Fcomplex(TEST_DIM1,TEST_DIM2); */
  /*   init_part(); run_part("create2Fcomplex", ArrayOps::check2Fcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* }  */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayOps::create3Int(); */
  /*   init_part(); run_part("create3Int", ArrayOps::check3Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayOps::create4Int(); */
  /*   init_part(); run_part("create4Int", ArrayOps::check4Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayOps::create5Int(); */
  /*   init_part(); run_part("create5Int", ArrayOps::check5Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayOps::create6Int(); */
  /*   init_part(); run_part("create6Int", ArrayOps::check6Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   iarray = ArrayOps::create7Int(); */
  /*   init_part(); run_part("create7Int", ArrayOps::check7Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<bool> barray; */
  /*   ArrayOps::makeInOutBool(barray, 218); */
  /*   init_part(); run_part("makeInOutBool", ArrayOps::checkBool(barray) == true); */
  /*   barray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<char> carray; */
  /*   ArrayOps::makeInOutChar(carray, 218); */
  /*   init_part(); run_part("makeInOutChar", ArrayOps::checkChar(carray) == true); */
  /*   carray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   ArrayOps::makeInOutInt(iarray, 218); */
  /*   init_part(); run_part("makeInOutInt", ArrayOps::checkInt(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int64_t> larray; */
  /*   ArrayOps::makeInOutLong(larray, 218); */
  /*   init_part(); run_part("makeInOutLong", ArrayOps::checkLong(larray) == true); */
  /*   larray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray; */
  /*   ArrayOps::makeInOutString(sarray, 218); */
  /*   init_part(); run_part("makeInOutString", ArrayOps::checkString(sarray) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayOps::makeInOutDouble(darray, 218); */
  /*   init_part(); run_part("makeInOutDouble", ArrayOps::checkDouble(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   ArrayOps::makeInOutFloat(farray, 218); */
  /*   init_part(); run_part("makeInOutFloat", ArrayOps::checkFloat(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayOps::makeInOutDcomplex(dcarray, 218); */
  /*   init_part(); run_part("makeInOutDcomplex", ArrayOps::checkDcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayOps::makeInOutFcomplex(fcarray, 218); */
  /*   init_part(); run_part("makeInOutFcomplex", ArrayOps::checkFcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray; */
  /*   ArrayOps::makeInOut2Int(iarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Int", ArrayOps::check2Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<double> darray; */
  /*   ArrayOps::makeInOut2Double(darray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Double", ArrayOps::check2Double(darray) == true); */
  /*   darray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<float> farray; */
  /*   ArrayOps::makeInOut2Float(farray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Float", ArrayOps::check2Float(farray) == true); */
  /*   farray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::dcomplex> dcarray; */
  /*   ArrayOps::makeInOut2Dcomplex(dcarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Dcomplex", ArrayOps::check2Dcomplex(dcarray) == true); */
  /*   dcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<sidl::fcomplex> fcarray; */
  /*   ArrayOps::makeInOut2Fcomplex(fcarray, TEST_DIM1, TEST_DIM2); */
  /*   init_part(); run_part("makeInOut2Fcomplex", ArrayOps::check2Fcomplex(fcarray) == true); */
  /*   fcarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayOps::makeInOut3Int(iarray); */
  /*   init_part(); run_part("makeInOut3Int", ArrayOps::check3Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayOps::makeInOut4Int(iarray); */
  /*   init_part(); run_part("makeInOut4Int", ArrayOps::check4Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayOps::makeInOut5Int(iarray); */
  /*   init_part(); run_part("makeInOut5Int", ArrayOps::check5Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayOps::makeInOut6Int(iarray); */
  /*   init_part(); run_part("makeInOut6Int", ArrayOps::check6Int(iarray) == true); */
  /*   iarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int> iarray; */
  /*   ArrayOps::makeInOut7Int(iarray); */
  /*   init_part(); run_part("makeInOut7Int", ArrayOps::check7Int(iarray) == true); */
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
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(objarray) == 0); */
  /*   for(int32_t i = 0; i < TEST_SIZE; i += 2) { */
  /*     objarray.set(i, ArrayOps::_create()); */
  /*   } */
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(objarray) == ((TEST_SIZE+1)/2)); */
  /*   sidl::array<ArrayTest::ArrayOps> sliced =  */
  /*     objarray.slice(1, numElem, start, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(sliced) == ((TEST_SIZE+1)/2)); */
  /*   sliced = objarray.slice(1, numElemTwo, startTwo, stride); */
  /*   init_part(); run_part("create1d", sliced._not_nil()); */
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(sliced) == 0); */
  /*   objarray.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(objarray) == ((TEST_SIZE+1)/2)); */
  /*   sliced.smartCopy(); */
  /*   init_part(); run_part("create1d", ArrayOps::checkObject(sliced) == 0); */

  /*   init_part(); run_part("createObjectNegOne", ArrayOps::createObject(-1)._is_nil()); */
  /*   init_part(); run_part("checkObjectNull", !ArrayOps::checkObject(NULL)); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */


  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<bool> ary; */
  /*   init_part(); run_part("createBoolNegOne", ArrayOps::createBool(-1)._is_nil()); */
  /*   ArrayOps::makeBool(-1, ary); */
  /*   init_part(); run_part("makeBoolNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<char> ary; */
  /*   init_part(); run_part("createCharNegOne", ArrayOps::createChar(-1)._is_nil()); */
  /*   ArrayOps::makeChar(-1, ary); */
  /*   init_part(); run_part("makeCharNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int32_t> ary; */
  /*   init_part(); run_part("createIntNegOne", ArrayOps::createInt(-1)._is_nil()); */
  /*   ArrayOps::makeInt(-1, ary); */
  /*   init_part(); run_part("makeIntNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<int64_t> ary; */
  /*   init_part(); run_part("createLongNegOne", ArrayOps::createLong(-1)._is_nil()); */
  /*   ArrayOps::makeLong(-1, ary); */
  /*   init_part(); run_part("makeLongNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<string> ary; */
  /*   init_part(); run_part("createStringNegOne", ArrayOps::createString(-1)._is_nil()); */
  /*   ArrayOps::makeString(-1, ary); */
  /*   init_part(); run_part("makeStringNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<double> ary; */
  /*   init_part(); run_part("createDoubleNegOne", ArrayOps::createDouble(-1)._is_nil()); */
  /*   ArrayOps::makeDouble(-1, ary); */
  /*   init_part(); run_part("makeDoubleNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<float> ary; */
  /*   init_part(); run_part("createFloatNegOne", ArrayOps::createFloat(-1)._is_nil()); */
  /*   ArrayOps::makeFloat(-1, ary); */
  /*   init_part(); run_part("makeFloatNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::dcomplex> ary; */
  /*   init_part(); run_part("createDcomplexNegOne", ArrayOps::createDcomplex(-1)._is_nil()); */
  /*   ArrayOps::makeDcomplex(-1, ary); */
  /*   init_part(); run_part("makeDcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ::sidl::array<sidl::fcomplex> ary; */
  /*   init_part(); run_part("createFcomplexNegOne", ArrayOps::createFcomplex(-1)._is_nil()); */
  /*   ArrayOps::makeFcomplex(-1, ary); */
  /*   init_part(); run_part("makeFcomplexNegOne", ary._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DoubleNegOne", ArrayOps::create2Double(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FcomplexNegOne", ArrayOps::create2Fcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2DcomplexNegOne", ArrayOps::create2Dcomplex(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2FloatNegOne", ArrayOps::create2Float(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */



  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   init_part(); run_part("create2IntNegOne", ArrayOps::create2Int(-1,-1)._is_nil()); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */

  /* { */
  /*   int32_t dimen, dimen2, type, type2 = 0; */
  /*   ::sidl::basearray garray, garrayret, garrayout, garrayinout; */
  /*   ::sidl::array<int32_t> iarray; */
  /*   magicNumber = clearstack(magicNumber); */
    
  /*   ArrayOps::checkGeneric(garray, dimen, type); */
  /*   init_part(); run_part("Generic array is still Null", garray._is_nil()); */
  /*   init_part(); run_part("NULL Generic array has no dimension", (dimen == 0)); */
  /*   init_part(); run_part("NULL Generic array has no type", (type == 0)); */

  /*   dimen = 1; */
  /*   type = sidl_int_array; */
  /*   garray = ArrayOps::createGeneric(dimen, type); */
  /*   init_part(); run_part("Generic (int) array is not Null", garray._not_nil()); */
  /*   init_part(); run_part("Generic (int) array has 1 dimension", (dimen == garray.dimen())); */
  /*   init_part(); run_part("Generic (int) array has int type", type == garray.arrayType()); */
    
  /*   ArrayOps::checkGeneric(garray, dimen2, type2); */
  /*   init_part(); run_part("checkGeneric (int) array has 1 dimension", (dimen == dimen2)); */
  /*   init_part(); run_part("checkGeneric (int) array has int type", (type == type2)); */

  /*   garrayret = ArrayOps::passGeneric(garray, iarray, */
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
  /* 	     ArrayOps::check2Int(iarray) == TRUE); */

  /* } */
  /* { */
  /*   int32_t* irarray = NULL; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayOps::createInt(TEST_SIZE); */

  /*   init_part(); run_part("createInt", iarray); */
  /*   init_part(); run_part("createInt", ArrayOps::checkInt(iarray) == TRUE); */

  /*   irarray = iarray.first();//->d_firstElement; */
  /*   init_part(); run_part("Check rarray int 1", ArrayOps::checkRarray1Int(irarray, TEST_SIZE) == TRUE); */
    
  /* } */

  /* { */
  /*   int32_t* irarray = NULL; */
  /*   int32_t n,m,o; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayOps::create3Int();//iarray = ArrayOps::create3Int(); */
  /*   init_part(); run_part("create3Int", ArrayOps::check3Int(iarray) == TRUE); */
  /*   iarray.ensure(3,sidl::column_major_order); */
  /*   irarray = iarray.first();//->d_firstElement; */
  /*   n = iarray.length(0); */
  /*   m = iarray.length(1); */
  /*   o = iarray.length(2); */
  /*   init_part(); run_part("Check rarray int 3", ArrayOps::checkRarray3Int(irarray, n,m,o) == TRUE); */
 
  /* } */

  /*   { */
  /*   int32_t* irarray = NULL; */
  /*   int32_t n,m,o,p,q,r,s; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<int32_t> iarray = ArrayOps::create7Int();//iarray = ArrayOps::create7Int(); */
  /*   init_part(); run_part("create3Int", ArrayOps::check7Int(iarray) == TRUE); */
  /*   iarray.ensure(7,sidl::column_major_order); */
  /*   irarray = iarray.first();//->d_firstElement; */
  /*   n = iarray.length(0); */
  /*   m = iarray.length(1); */
  /*   o = iarray.length(2); */
  /*   p = iarray.length(3); */
  /*   q = iarray.length(4); */
  /*   r = iarray.length(5); */
  /*   s = iarray.length(6); */
  /*   init_part(); run_part("Check rarray int 7", ArrayOps::checkRarray7Int(irarray, n,m,o,p,q,r,s) == TRUE); */

  /* } */


  {
    var irarray: [0..TEST_SIZE-1] int(32);
    ArrayOps.initRarray1Int(irarray, TEST_SIZE);
    init_part(); run_part("Check rarray int 1", ArrayOps::checkRarray1Int(irarray, TEST_SIZE) == TRUE);
  }

  /* { */
  /*   int32_t irarray[24]; //2*3*4  */
  /*   int32_t n=2, m=3 , o=4; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ArrayOps::initRarray3Int(irarray, n,m,o); */
  /*   init_part(); run_part("Check rarray int 3", ArrayOps::checkRarray3Int(irarray, n,m,o) == TRUE); */

  /* } */

  /* { */
  /*   int32_t irarray[432]; //2*2*2*2*3*3*3  */
  /*   int32_t n=2, m=2 , o=2, p=2, q=3,r=3,s=3; */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ArrayOps::initRarray7Int(irarray, n,m,o,p,q,r,s); */
  /*   init_part(); run_part("Check rarray int 7", ArrayOps::checkRarray7Int(irarray, n,m,o,p,q,r,s) == TRUE); */

  /* } */


  /* { */
  /*   double drarray[TEST_SIZE];  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ArrayOps::initRarray1Double(drarray, TEST_SIZE); */
  /*   init_part(); run_part("Check rarray double 1", ArrayOps::checkRarray1Double(drarray, TEST_SIZE) == TRUE); */
  /* } */

  /* { */
  /*   struct sidl_dcomplex dcrarray[TEST_SIZE];  */
  /*   magicNumber = clearstack(magicNumber); */
  /*   ArrayOps::initRarray1Dcomplex(dcrarray, TEST_SIZE); */
  /*   init_part(); run_part("Check rarray Dcomplex 1", ArrayOps::checkRarray1Dcomplex(dcrarray, TEST_SIZE) == TRUE); */
  /* } */

  /* { */
  /*   int32_t n = 3, m = 3, o = 2, a[9], b[6], x[6]; */
  /*   int i = 0; */
  /*   for(i = 0; i < 9; ++i) { */
  /*     a[i]=i; */
  /*   } */
  /*   for(i = 0; i < 6; ++i) { */
  /*     b[i]=i; */
  /*   } */
  /*   ArrayOps::matrixMultiply(a,b,x,n,m,o); */
  /*   init_part(); run_part("Check Matrix Multiply", ArrayOps::checkMatrixMultiply(a,b,x,n,m,o) == TRUE); */
	
  /* } */
  
  /* { */
  /*   magicNumber = clearstack(magicNumber); */
  /*   sidl::array<string> sarray = ArrayOps::create2String(12,13); */
  /*   init_part(); run_part("createString", sarray._not_nil()); */
  /*   init_part(); run_part("createString", ArrayOps::check2String(sarray) == true); */
  /*   sarray.deleteRef(); */
  /*   magicNumber = clearstack(magicNumber); */
  /* } */


  tracker.close();
  return 0;
}
