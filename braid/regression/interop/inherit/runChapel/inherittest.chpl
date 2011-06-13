// 
// File:        inherittest.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Simple test on the InheritTest static methods
// 
use Inherit;
use synch;
use sidl;

var part_no: int = 0;
var tracker: synch.RegOut = synch.RegOut_static.getInstance();
var magicNumber = 13;
//tracker.setExpectations(76);
tracker.setExpectations(-1);

proc init_part(name: string) 
{
  part_no += 1;
  tracker.startPart(part_no);
  tracker.writeComment("Method "+name);
  magicNumber = clearstack(magicNumber);
}


proc run_part(result: string, template: string)
{
  var r: ResultType;
  tracker.writeComment("should return "+result);
  if (result == template) then
    r = ResultType.PASS;
  else 
    r = ResultType.FAIL;
  tracker.endPart(part_no, r);
  tracker.writeComment("Method returned "+result);
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

{ 
  var c = new Inherit.C();
  tracker.writeComment("Class C:");
  init_part("c.c()"); run_part(c.c(), "C.c");
}

/*   {  */
/*     Inherit::D d = makeDObject(); */
/*     tracker.writeComment("Class D: inheritance of interface A"); */
/*     INHERITTEST(d.a(), "D.a"); */
/*     INHERITTEST(d.d(), "D.d"); */


/*     tracker.writeComment("Class D: via interface A"); */
/*     Inherit::A a(d); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting D to interface A"); */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(a.a(), "D.a"); */
/*     } */

/*     tracker.writeComment("Class D2: via interface A"); */
/*     Inherit::D d2 = ::sidl::babel_cast<Inherit::D>(a); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting A to interface D2"); */
/*     if ( !d2 ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(d2.d(), "D.d"); */
/*     } */

/*   } */

/*   {  */
/*     Inherit::E e = makeEObject();  */
/*     tracker.writeComment("Class E: inheritance of class C"); */
/*     INHERITTEST(e.c(), "C.c"); */
/*     INHERITTEST(e.e(), "E.e"); */

/*     tracker.writeComment("Class E: via class C (C.c not overridden)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting E to class C"); */
/*     Inherit::C c = e; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(c.c(), "C.c"); */
/*     } */
/*   } */

/*   {  */
/*     Inherit::E2 e2 = makeE2Object();  */
/*     tracker.writeComment("Class E2: inheritance of class C"); */
/*     INHERITTEST(e2.c(), "E2.c"); */
/*     INHERITTEST(e2.e(), "E2.e"); */

/*     tracker.writeComment("Class E2: via class C (C.c overridden)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting E2 to class C"); */
/*     Inherit::C c = e2; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(c.c(), "E2.c"); */
/*     } */

/*     INHERITTEST(Inherit::E2::m(), "E2.m"); */
/*   } */

/*   {  */
/*     Inherit::F f = makeFObject(); */
/*     tracker.writeComment("Class F: Multiple inheritance (no overriding)"); */
/*     INHERITTEST(f.a(), "F.a"); */
/*     INHERITTEST(f.b(), "F.b"); */
/*     INHERITTEST(f.c(), "C.c"); */
/*     INHERITTEST(f.f(), "F.f"); */
    
/*     tracker.writeComment("Class F: via interface A") ; */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to class A"); */
/*     Inherit::A a = f; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(a.a(), "F.a"); */
/*     } */

/*     tracker.writeComment("Class F: via interface B"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to interface B"); */
/*     Inherit::B b = f; */
/*     if ( !b ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(b.b(), "F.b"); */
/*     } */


/*     tracker.writeComment("Class F: via class C (no overloading of C.c)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to class C"); */
/*     Inherit::C c = f; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(c.c(), "C.c"); */
/*     }  */
/*   } */

/*   {  */
/*     Inherit::F2 f2 = makeF2Object(); */
/*     tracker.writeComment("Class F2: Multiple inheritance (overrides C.c)"); */
/*     INHERITTEST(f2.a(), "F2.a"); */
/*     INHERITTEST(f2.b(), "F2.b"); */
/*     INHERITTEST(f2.c(), "F2.c"); */
/*     INHERITTEST(f2.f(), "F2.f"); */
    
/*     tracker.writeComment("Class F2: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to interface A"); */
/*     Inherit::A a = f2; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(a.a(), "F2.a"); */
/*     } */

/*     tracker.writeComment("Class F2: via interface B"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to interface B"); */
/*     Inherit::B b = f2; */
/*     if ( !b ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(b.b(), "F2.b"); */
/*     } */

/*     tracker.writeComment("Class F2: via class C (overloads C.c)") ; */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to class C"); */
/*     Inherit::C c = f2; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(c.c(), "F2.c"); */
/*     }  */
/*   } */

/*   {  */
/*     Inherit::G g = makeGObject(); */

/*     tracker.writeComment("Class G: indirect multiple inheritance ( no overloads)"); */
/*     INHERITTEST(g.a(), "D.a"); */
/*     INHERITTEST(g.d(), "D.d"); */
/*     INHERITTEST(g.g(), "G.g"); */

    
/*     tracker.writeComment("Class G: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G to interface A"); */
/*     Inherit::A a = g; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(a.a(), "D.a"); */
/*     } */

/*     tracker.writeComment("Class G: via class D"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G to class D"); */
/*     Inherit::D d = g; */
/*     if ( !d ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(d.a(), "D.a"); */
/*       INHERITTEST(d.d(), "D.d"); */
/*     } */

/*   } */
  
/*   {  */
/*     Inherit::G2 g2 = makeG2Object(); */

/*     tracker.writeComment("Class G2: indirect multiple inheritance (overloads)"); */
/*     INHERITTEST(g2.a(), "G2.a"); */
/*     INHERITTEST(g2.d(), "G2.d"); */
/*     INHERITTEST(g2.g(), "G2.g"); */

    
/*     tracker.writeComment("Class G2: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G2 to interface A"); */
/*     Inherit::A a = g2; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(a.a(), "G2.a"); */
/*     } */

/*     tracker.writeComment("Class G2: via class D"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G2 to class D"); */
/*     Inherit::D d = g2; */
/*     if ( !d ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       INHERITTEST(d.a(), "G2.a"); */
/*       INHERITTEST(d.d(), "G2.d"); */
/*     } */

/*   } */

/*   { */
/*     Inherit::I i = makeIObject(); */
/*     tracker.writeComment("Class I: implements abstract class H that implements A"); */
/*     INHERITTEST(i.a(), "I.a"); */
/*     INHERITTEST(i.h(), "I.h"); */
    
/*     tracker.writeComment("Class I: via interface A"); */
/*     Inherit::A a = i; */
/*     INHERITTEST(a.a(), "I.a"); */
    
/*     tracker.writeComment("Class I: via abstract class H"); */
/*     Inherit::H h = i; */
/*     INHERITTEST(h.a(), "I.a"); */
/*     INHERITTEST(h.h(), "I.h"); */
/*   } */

/*   { */
/*     Inherit::J j = makeJObject(); */
/*     tracker.writeComment("\nClass J: implements A and B, extends E. Calls super of E and C\n"); */
/*     INHERITTEST(j.a(), "J.a"); */
/*     INHERITTEST(j.b(), "J.b"); */
/*     INHERITTEST(j.j(), "J.j"); */
/*     INHERITTEST(j.c(), "J.E2.c"); */
/*     INHERITTEST(j.e(), "J.E2.e"); */
    
/*     INHERITTEST(Inherit::J::m(), "E2.m"); */
/*   } */

/*   { */
/*     Inherit::K k = makeKObject(); */
/*     tracker.writeComment("Class K: implements A2, extends H."); */
/*     INHERITTEST(k.a(), "K.a"); */
/*     INHERITTEST(k.a(0), "K.a2"); */
/*     INHERITTEST(k.h(), "K.h"); */
/*     INHERITTEST(k.k(), "K.k"); */
    
/*     tracker.writeComment("Class K: via interface A"); */
/*     Inherit::A a = k; */
/*     INHERITTEST(a.a(), "K.a"); */
   
/*     tracker.writeComment("Class K: via interface A2"); */
/*     Inherit::A2 a2 = k; */
/*     INHERITTEST(a2.a(0), "K.a2"); */
    
/*     tracker.writeComment("Class K: via abstract class H"); */
/*     Inherit::H h = k; */
/*     INHERITTEST(h.a(), "K.a"); */
/*     INHERITTEST(h.h(), "K.h"); */
/*   } */

/*   { */
/*     Inherit::L l = makeLObject(); */
/*     tracker.writeComment("Class L: implements A, A2."); */
/*     INHERITTEST(l.a(), "L.a"); */
/*     INHERITTEST(l.a(0), "L.a2"); */
/*     INHERITTEST(l.l(), "L.l"); */
    
/*     tracker.writeComment("Class L: via interface A"); */
/*     Inherit::A a = l; */
/*     INHERITTEST(a.a(), "L.a"); */
   
/*     tracker.writeComment("Class L: via interface A2"); */
/*     Inherit::A2 a2 = l; */
/*     INHERITTEST(a2.a(0), "L.a2"); */
    
/*   } */

tracker.close();

