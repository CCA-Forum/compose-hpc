// 
// File:        stringstests.chpl
// Copyright:   (c) 2012 Lawrence Livermore National Security, LLC
// Description: Test string interoperability
// 
use s;
use synch;

config var bindir = "gantlet compatibility";

var failed: bool = false;
var part_no: int(32) = 0;
var sidl_ex: BaseInterface = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);
extern proc IS_NULL(in aRef): bool;

proc init_part() {
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Part " + part_no, sidl_ex);
}

proc run_part(result: bool) {
  run_part("", result);
}

proc run_part(msg: string, result: bool) {
  if (msg.length > 0) {
    tracker.writeComment(msg, sidl_ex);
  }
  var r: ResultType;
  if (result) then
    r = ResultType.PASS;
  else {
    r = ResultType.FAIL;
    failed = true;
  }
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("End of part " + part_no, sidl_ex);
}
/* #include "s_StructTest.hxx" */
/* #include <sstream> */
/* #include <iostream> */
/* #include <cstring> */
/* #include <string.h> */
/* #include <cstdio> */

/* #ifdef WITH_RMI */
/* #include "sidl_rmi_ProtocolFactory.hxx" */
/* #endif */

/* using namespace std; */
/* #include "synch.hxx" */

/* #define MYASSERT( AAA ) \ */
/*   tracker.startPart(++part_no);\ */
/*   tracker.writeComment(#AAA); \ */
/*   if ( AAA ) result = synch::ResultType_PASS; \ */
/*   else result = synch::ResultType_FAIL;  \ */
/*   tracker.endPart(part_no, result); */

proc
initSimple(s1: s.s_Simple)
{
  s1.d_bool = true;
  s1.d_char = '3';
  s1.d_dcomplex.re = 3.14:real(64);
  s1.d_dcomplex.im = 3.14:real(64);
  s1.d_fcomplex.re = 3.1:real(32);
  s1.d_fcomplex.im = 3.1:real(32);
  s1.d_double = 3.14:real(64);
  s1.d_float = 3.1:real(32);
  s1.d_int = 3;
  s1.d_long = 3;
  s1.d_opaque = 0;
  s1.d_enum = s.Color.blue;
}

proc
initHard(h: s.s_Hard) {
}
/*   ::sidl::array< ::std::string> str = ::sidl::array< ::std::string> ::create1d(1); */
/*   str.set(0,"Three"); */
/*   h.set_d_string(str); */
/*   h.set_d_object(::sidl::BaseClass::_create()); */
/*   h.set_d_interface(h.get_d_object()); */
/*   ::sidl::array<double> da = ::sidl::array<double> ::create1d(3); */
/*   da.set(0,1.0); */
/*   da.set(1,2.0); */
/*   da.set(2,3.0); */
/*   h.set_d_array(da); */
/*   ::sidl::array< ::sidl::BaseClass > oa =  */
/*       ::sidl::array< ::sidl::BaseClass > :: create1d(3); */
/*   oa.set(0, ::sidl::BaseClass::_create()); */
/*   oa.set(1, ::sidl::BaseClass::_create()); */
/*   oa.set(2, ::sidl::BaseClass::_create()); */
/*   h.set_d_objectArray(oa); */
/* } */


proc
initRarrays(r: s.s_Rarrays) {
}
/*   const int N=3; */
/*   int i; */

/*   r.d_rarrayRaw = new double[N]; */

/*   r.d_int=N; */
/*   for (i=0; i<N;i++){ */
/*     r.d_rarrayRaw[i]=(double)(i+1); */
/*     r.d_rarrayFix[i]=(double)((i+1)*5); */
/*   } */
/* } */

proc
initCombined(c: s.s_Combined)
{
/*   initSimple(c.get_d_simple()); */
/*   initHard(c.get_d_hard()); */
}

proc
checkSimple(s1: s.s_Simple) : bool
{
  const eps : real = 1.E-6;

  return ((s1.d_bool &&
           (s1.d_char == '3') &&
           (abs(s1.d_dcomplex.re - 3.14) < eps) &&
           (abs(s1.d_dcomplex.im - 3.14) < eps) &&
           (abs(s1.d_double - 3.14) < eps) &&
           (abs(s1.d_fcomplex.re - 3.1:real(32)) < eps) &&
           (abs(s1.d_fcomplex.im - 3.1:real(32)) < eps) &&
           (abs(s1.d_float - 3.1:real(32)) < eps) &&
           (s1.d_int == 3) &&
           (s1.d_long == 3) &&
           (IS_NULL(s1.d_opaque)) &&
           (s1.d_enum == s.Color.blue)));
}

proc
checkSimpleInv(s1: s.s_Simple) : bool
{
  const eps : real = 1.E-6;

  return ((!s1.d_bool &&
           (s1.d_char == '3') &&
           (abs(s1.d_dcomplex.re - 3.14) < eps) &&
           (abs(s1.d_dcomplex.im + 3.14) < eps) &&
           (abs(s1.d_double + 3.14) < eps) &&
           (abs(s1.d_fcomplex.re - 3.1:real(32)) < eps) &&
           (abs(s1.d_fcomplex.im + 3.1:real(32)) < eps) &&
           (abs(s1.d_float + 3.1:real(32)) < eps) &&
           (s1.d_int == -3) &&
           (s1.d_long == -3) &&
           (IS_NULL(s1.d_opaque)) &&
           (s1.d_enum == s.Color.red)));
}

proc
checkHard(h: s.s_Hard) : bool
{
//   //bool result = (h.get_d_string() == "Three");
  var result : bool = false; //h.d_string != "";
//   if (result) {
//     ::sidl::array<string> str = h.get_d_string();
//     result = result && (str.dimen() == 1);
//     result = result && (str.length(0) == 1);
//     result = result && ("Three" == str[0]);
//   }
//   result = result && h.get_d_object()._not_nil();
//   result = result && h.get_d_interface()._not_nil();
//   if (result) {
//     result = result && h.get_d_object().
//       isSame(h.get_d_interface());
//   }
//   result = result && h.get_d_array()._not_nil();
//   if (result) {
//     ::sidl::array<double> da = h.get_d_array();
//     result = result && (da.dimen() == 1);
//     result = result && (da.length(0) == 3);
//     result = result && (da[0] == 1.0);
//     result = result && (da[1] == 2.0);
//     result = result && (da[2] == 3.0);
//   }
//   result = result && h.get_d_objectArray()._not_nil();
//   if (result) {
//     ::sidl::array< ::sidl::BaseClass > oa = h.get_d_objectArray();
//     result = result && (oa.dimen() == 1);
//     result = result && (oa.length(0) == 3);
//     result = result && oa[0]._not_nil() && oa[0].isType("sidl.BaseClass");
//     result = result && oa[1]._not_nil() && oa[1].isType("sidl.BaseClass");
//     result = result && oa[2]._not_nil() && oa[2].isType("sidl.BaseClass");
//   }
   return result;
}

proc
checkHardInv(h: s.s_Hard) :bool
{
  var result : bool = false; //h.d_string != "";
  // if (result) {
  //   ::sidl::array<string> str = h.get_d_string();
  //   result = result && (str.dimen() == 1);
  //   result = result && (str.length(0) == 1);
  //   result = result && ("three" == str[0]);
  // }
  // result = result && h.get_d_object()._not_nil();
  // result = result && h.get_d_interface()._not_nil();
  // if (result) {
  //   result = result && !(h.get_d_object().
  // 			 isSame(h.get_d_interface()));
  // }
  // result = result && h.get_d_array()._not_nil();
  // if (result) {
  //   const eps : double = 1.E-5;
  //   ::sidl::array<double> da = h.get_d_array();
  //   result = result && (da.dimen() == 1);
  //   result = result && (da.length(0) == 3);
  //   result = result && (abs(da[0] - 3.0) < eps);
  //   result = result && (abs(da[1] - 2.0) < eps);
  //   result = result && (abs(da[2] - 1.0) < eps);
  // }
  // result = result && h.get_d_objectArray()._not_nil();
  // if (result) {
  //   ::sidl::array< ::sidl::BaseClass > oa = h.get_d_objectArray();
  //   result = result && (oa.dimen() == 1);
  //   result = result && (oa.length(0) == 3);
  //   result = result && oa[0]._not_nil() && oa[0].isType("sidl.BaseClass");
  //   result = result && oa[1]._is_nil();
  //   result = result && oa[2]._not_nil() && oa[2].isType("sidl.BaseClass");
  // }
  return result;
}

proc
checkRarrays(r: s.s_Rarrays) : bool
{
  const eps : real = 1.E-5;
  var result = false; //(r.d_rarrayRaw != NULL);
  // if (result) {
  //   result = result && (abs(r.d_rarrayRaw[0] - 1.0) < eps);
  //   result = result && (abs(r.d_rarrayRaw[1] - 2.0) < eps);
  //   result = result && (abs(r.d_rarrayRaw[2] - 3.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[0] - 5.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[1] - 10.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[2] - 15.0) < eps);
  // }
  return result;
}

proc
checkRarraysInv(r: s.s_Rarrays) 
 :bool
{
  const eps : real = 1.E-5;
  var result = false; //(r.d_rarrayRaw != NULL);
  // if (result) {
  //   result = result && (abs(r.d_rarrayRaw[0] - 3.0) < eps);
  //   result = result && (abs(r.d_rarrayRaw[1] - 2.0) < eps);
  //   result = result && (abs(r.d_rarrayRaw[2] - 1.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[0] - 15.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[1] - 10.0) < eps);
  //   result = result && (abs(r.d_rarrayFix[2] - 5.0) < eps);
  // }
  return result;
}

proc 
deleteRarrays(r: s.s_Rarrays) {

  // delete [] r.d_rarrayRaw;

}

proc
checkCombined(c: s.s_Combined) : bool
{
  return checkSimple(c.d_simple) && checkHard(c.d_hard);
}

proc
checkCombinedInv(c: s.s_Combined) : bool
{
   return checkSimpleInv(c.d_simple) && checkHardInv(c.d_hard);
}

tracker.setExpectations(36, sidl_ex);
var test:StructTest = s.StructTest_static.create_StructTest(sidl_ex);
//MYASSERT(test._not_nil());

{
  var e1, e2, e3, e4 : s.s_Empty;
  /* ostringstream buf; */
  /* buf << "sizeof(s.s_Empty) == " << sizeof(s.s_Empty); */
  /* tracker.writeComment(buf.str()); */
  e1 = test.returnEmpty(sidl_ex);
  init_part(); run_part( test.passinEmpty(e1, sidl_ex) );
  init_part(); run_part( test.passoutEmpty(e1, sidl_ex) );
  init_part(); run_part( test.passoutEmpty(e2, sidl_ex) );
  init_part(); run_part( test.passinoutEmpty(e2, sidl_ex) );
  init_part(); run_part( test.passoutEmpty(e3, sidl_ex) );
  e4 = test.passeverywhereEmpty(e1, e2, e3, sidl_ex) ;
  init_part(); run_part( true );
}

{
  var  s1, s2, s3, s4 : s.s_Simple;
/*   ostringstream buf; */
/*   buf << "sizeof(s.s_Simple) == " << sizeof(s.s_Simple); */
/*   tracker.writeComment(buf.str()); */
  s1 = test.returnSimple(sidl_ex);
  init_part(); run_part( checkSimple(s1) );
  init_part(); run_part( test.passinSimple(s1, sidl_ex) );
  init_part(); run_part( test.passoutSimple(s1, sidl_ex) );
  init_part(); run_part( test.passoutSimple(s2, sidl_ex) );
  init_part(); run_part( test.passinoutSimple(s2, sidl_ex) );
  init_part(); run_part( checkSimpleInv(s2) );
  init_part(); run_part( test.passoutSimple(s3, sidl_ex) );
  s4 = test.passeverywhereSimple(s1, s2, s3, sidl_ex);
  init_part(); run_part( checkSimple(s4) && checkSimple(s2) && checkSimpleInv(s3) );
}

/* //some elements in s.Hard can't be passed as they are not */
/* //serializable, so skip these tests for RMI */

/* if (!withRMI()) { */
{
  var h1, h2, h3, h4 : s.s_Hard;
  // ostringstream buf;
  // buf << "sizeof(s.s_Hard) == " << sizeof(s.s_Hard);
  // tracker.writeComment(buf.str(, sidl_ex) );
  h1 = test.returnHard(sidl_ex);
  init_part(); run_part( checkHard(h1) );
  init_part(); run_part( test.passinHard(h1, sidl_ex) );
  init_part(); run_part( test.passoutHard(h1, sidl_ex) );
  init_part(); run_part( test.passoutHard(h2, sidl_ex) );
  init_part(); run_part( test.passinoutHard(h2, sidl_ex) );
  init_part(); run_part( checkHardInv(h2) );
  init_part(); run_part( test.passoutHard(h3, sidl_ex) );
  h4 = test.passeverywhereHard(h1, h2, h3, sidl_ex);
  init_part(); run_part( checkHard(h4) && checkHard(h2) && checkHardInv(h3) );
}

{
  var r1, r2: s.s_Rarrays;
//   ostringstream buf;
//   buf << "sizeof(s.s_Rarrays) == " << sizeof(s.s_Rarrays)
// << " sizeof(struct s_Rarrays__data) == "
// << sizeof(struct s_Rarrays__data);
//   tracker.writeComment(buf.str(, sidl_ex) );
  initRarrays(r1);
  init_part(); run_part( test.passinRarrays(r1, sidl_ex) );
  init_part(); run_part( test.passinoutRarrays(r1, sidl_ex) );
  init_part(); run_part( checkRarraysInv(r1) );
  deleteRarrays(r1);
  initRarrays(r1);
  initRarrays(r2);
  init_part(); run_part( test.passeverywhereRarrays(r1, r2, sidl_ex) );
  init_part(); run_part( checkRarrays(r1) && checkRarraysInv(r2) );
  deleteRarrays(r1);
  deleteRarrays(r2);
}

{
  var c0, c1, c2, c3, c4: s.s_Combined;
//   ostringstream buf;
//   buf << "sizeof(s.s_Combined) == " << sizeof(s.s_Combined)
// << " sizeof(struct s_Combined__data) == "
// << sizeof(struct s_Combined__data);
//   tracker.writeComment(buf.str(, sidl_ex) );
  initCombined(c0);
  init_part(); run_part( checkCombined(c0) );
  c1 = test.returnCombined(sidl_ex);
  init_part(); run_part( checkCombined(c1) );
  init_part(); run_part( test.passinCombined(c1, sidl_ex) );
  init_part(); run_part( test.passoutCombined(c1, sidl_ex) );
  init_part(); run_part( test.passoutCombined(c2, sidl_ex) );
  init_part(); run_part( test.passinoutCombined(c2, sidl_ex) );
  init_part(); run_part( checkCombinedInv(c2) );
  init_part(); run_part( test.passoutCombined(c3, sidl_ex) );
  c4 = test.passeverywhereCombined(c1, c2, c3, sidl_ex);
  init_part(); run_part( checkCombined(c4) && checkCombined(c2) && checkCombinedInv(c3) );
}
tracker.close(sidl_ex);

if (failed) then
  exit(1);
