// 
// File:        inherittest.chpl
// Copyright:   (c) 2011 Lawrence Livermore National Security, LLC
// Description: Simple test on the InheritTest static methods
// 
use Inherit;
use synch;
use sidl;

config var bindir = "gantlet compatibility";

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

  {
    var d = new Inherit.D();
    tracker.writeComment("Class D: inheritance of interface A");
    init_part("d.a()"); run_part(d.a(), "D.a");
    init_part("d.d()"); run_part(d.d(), "D.d");

    tracker.writeComment("Class D: via interface A");
    var a = new Inherit.A(d.cast_Inherit_A());
    part_no += 1;
    tracker.startPart(part_no);
    tracker.writeComment("Casting D to interface A");
    if ( a == nil) {
      tracker.endPart(part_no, synch.ResultType.FAIL);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS);
      init_part("a.a()"); run_part(a.a(), "D.a");
    }
  }
/*     tracker.writeComment("Class D2: via interface A"); */
/*     Inherit::D d2 = ::sidl::babel_cast<Inherit::D>(a); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting A to interface D2"); */
/*     if ( !d2 ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("d2.d()"); run_part(d2.d(), "D.d"); */
/*     } */

/*   } */

/*   {  */
/*     Inherit::E e = makeEObject();  */
/*     tracker.writeComment("Class E: inheritance of class C"); */
/*     init_part("e.c()"); run_part(e.c(), "C.c"); */
/*     init_part("e.e()"); run_part(e.e(), "E.e"); */

/*     tracker.writeComment("Class E: via class C (C.c not overridden)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting E to class C"); */
/*     Inherit::C c = e; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("c.c()"); run_part(c.c(), "C.c"); */
/*     } */
/*   } */

/*   {  */
/*     Inherit::E2 e2 = makeE2Object();  */
/*     tracker.writeComment("Class E2: inheritance of class C"); */
/*     init_part("e2.c()"); run_part(e2.c(), "E2.c"); */
/*     init_part("e2.e()"); run_part(e2.e(), "E2.e"); */

/*     tracker.writeComment("Class E2: via class C (C.c overridden)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting E2 to class C"); */
/*     Inherit::C c = e2; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("c.c()"); run_part(c.c(), "E2.c"); */
/*     } */

/*     init_part("Inherit::E2::m()"); run_part(Inherit::E2::m(), "E2.m"); */
/*   } */

/*   {  */
/*     Inherit::F f = makeFObject(); */
/*     tracker.writeComment("Class F: Multiple inheritance (no overriding)"); */
/*     init_part("f.a()"); run_part(f.a(), "F.a"); */
/*     init_part("f.b()"); run_part(f.b(), "F.b"); */
/*     init_part("f.c()"); run_part(f.c(), "C.c"); */
/*     init_part("f.f()"); run_part(f.f(), "F.f"); */
    
/*     tracker.writeComment("Class F: via interface A") ; */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to class A"); */
/*     Inherit::A a = f; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("a.a()"); run_part(a.a(), "F.a"); */
/*     } */

/*     tracker.writeComment("Class F: via interface B"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to interface B"); */
/*     Inherit::B b = f; */
/*     if ( !b ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("b.b()"); run_part(b.b(), "F.b"); */
/*     } */


/*     tracker.writeComment("Class F: via class C (no overloading of C.c)"); */
/*     tracker.startPart(++part_no); */
/*     tracker.writeComment("Casting F to class C"); */
/*     Inherit::C c = f; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("c.c()"); run_part(c.c(), "C.c"); */
/*     }  */
/*   } */

/*   {  */
/*     Inherit::F2 f2 = makeF2Object(); */
/*     tracker.writeComment("Class F2: Multiple inheritance (overrides C.c)"); */
/*     init_part("f2.a()"); run_part(f2.a(), "F2.a"); */
/*     init_part("f2.b()"); run_part(f2.b(), "F2.b"); */
/*     init_part("f2.c()"); run_part(f2.c(), "F2.c"); */
/*     init_part("f2.f()"); run_part(f2.f(), "F2.f"); */
    
/*     tracker.writeComment("Class F2: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to interface A"); */
/*     Inherit::A a = f2; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("a.a()"); run_part(a.a(), "F2.a"); */
/*     } */

/*     tracker.writeComment("Class F2: via interface B"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to interface B"); */
/*     Inherit::B b = f2; */
/*     if ( !b ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("b.b()"); run_part(b.b(), "F2.b"); */
/*     } */

/*     tracker.writeComment("Class F2: via class C (overloads C.c)") ; */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting F2 to class C"); */
/*     Inherit::C c = f2; */
/*     if ( !c ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("c.c()"); run_part(c.c(), "F2.c"); */
/*     }  */
/*   } */

/*   {  */
/*     Inherit::G g = makeGObject(); */

/*     tracker.writeComment("Class G: indirect multiple inheritance ( no overloads)"); */
/*     init_part("g.a()"); run_part(g.a(), "D.a"); */
/*     init_part("g.d()"); run_part(g.d(), "D.d"); */
/*     init_part("g.g()"); run_part(g.g(), "G.g"); */

    
/*     tracker.writeComment("Class G: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G to interface A"); */
/*     Inherit::A a = g; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("a.a()"); run_part(a.a(), "D.a"); */
/*     } */

/*     tracker.writeComment("Class G: via class D"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G to class D"); */
/*     Inherit::D d = g; */
/*     if ( !d ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("d.a()"); run_part(d.a(), "D.a"); */
/*       init_part("d.d()"); run_part(d.d(), "D.d"); */
/*     } */

/*   } */
  
/*   {  */
/*     Inherit::G2 g2 = makeG2Object(); */

/*     tracker.writeComment("Class G2: indirect multiple inheritance (overloads)"); */
/*     init_part("g2.a()"); run_part(g2.a(), "G2.a"); */
/*     init_part("g2.d()"); run_part(g2.d(), "G2.d"); */
/*     init_part("g2.g()"); run_part(g2.g(), "G2.g"); */

    
/*     tracker.writeComment("Class G2: via interface A"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G2 to interface A"); */
/*     Inherit::A a = g2; */
/*     if ( !a ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("a.a()"); run_part(a.a(), "G2.a"); */
/*     } */

/*     tracker.writeComment("Class G2: via class D"); */
/*     tracker.startPart(++part_no); */
/* tracker.writeComment("Casting G2 to class D"); */
/*     Inherit::D d = g2; */
/*     if ( !d ) {  */
/*       tracker.endPart(part_no, synch::ResultType_FAIL); */
/*     } else {  */
/*       tracker.endPart(part_no, synch::ResultType_PASS); */
/*       init_part("d.a()"); run_part(d.a(), "G2.a"); */
/*       init_part("d.d()"); run_part(d.d(), "G2.d"); */
/*     } */

/*   } */

/*   { */
/*     Inherit::I i = makeIObject(); */
/*     tracker.writeComment("Class I: implements abstract class H that implements A"); */
/*     init_part("i.a()"); run_part(i.a(), "I.a"); */
/*     init_part("i.h()"); run_part(i.h(), "I.h"); */
    
/*     tracker.writeComment("Class I: via interface A"); */
/*     Inherit::A a = i; */
/*     init_part("a.a()"); run_part(a.a(), "I.a"); */
    
/*     tracker.writeComment("Class I: via abstract class H"); */
/*     Inherit::H h = i; */
/*     init_part("h.a()"); run_part(h.a(), "I.a"); */
/*     init_part("h.h()"); run_part(h.h(), "I.h"); */
/*   } */

/*   { */
/*     Inherit::J j = makeJObject(); */
/*     tracker.writeComment("\nClass J: implements A and B, extends E. Calls super of E and C\n"); */
/*     init_part("j.a()"); run_part(j.a(), "J.a"); */
/*     init_part("j.b()"); run_part(j.b(), "J.b"); */
/*     init_part("j.j()"); run_part(j.j(), "J.j"); */
/*     init_part("j.c()"); run_part(j.c(), "J.E2.c"); */
/*     init_part("j.e()"); run_part(j.e(), "J.E2.e"); */
    
/*     init_part("Inherit::J::m()"); run_part(Inherit::J::m(), "E2.m"); */
/*   } */

/*   { */
/*     Inherit::K k = makeKObject(); */
/*     tracker.writeComment("Class K: implements A2, extends H."); */
/*     init_part("k.a()"); run_part(k.a(), "K.a"); */
/*     init_part("k.a(0)"); run_part(k.a(0), "K.a2"); */
/*     init_part("k.h()"); run_part(k.h(), "K.h"); */
/*     init_part("k.k()"); run_part(k.k(), "K.k"); */
    
/*     tracker.writeComment("Class K: via interface A"); */
/*     Inherit::A a = k; */
/*     init_part("a.a()"); run_part(a.a(), "K.a"); */
   
/*     tracker.writeComment("Class K: via interface A2"); */
/*     Inherit::A2 a2 = k; */
/*     init_part("a2.a(0)"); run_part(a2.a(0), "K.a2"); */
    
/*     tracker.writeComment("Class K: via abstract class H"); */
/*     Inherit::H h = k; */
/*     init_part("h.a()"); run_part(h.a(), "K.a"); */
/*     init_part("h.h()"); run_part(h.h(), "K.h"); */
/*   } */

/*   { */
/*     Inherit::L l = makeLObject(); */
/*     tracker.writeComment("Class L: implements A, A2."); */
/*     init_part("l.a()"); run_part(l.a(), "L.a"); */
/*     init_part("l.a(0)"); run_part(l.a(0), "L.a2"); */
/*     init_part("l.l()"); run_part(l.l(), "L.l"); */
    
/*     tracker.writeComment("Class L: via interface A"); */
/*     Inherit::A a = l; */
/*     init_part("a.a()"); run_part(a.a(), "L.a"); */
   
/*     tracker.writeComment("Class L: via interface A2"); */
/*     Inherit::A2 a2 = l; */
/*     init_part("a2.a(0)"); run_part(a2.a(0), "L.a2"); */
    
/*   } */

tracker.close();

