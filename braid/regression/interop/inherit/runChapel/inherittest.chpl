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
var sidl_ex: SidlBaseException = nil;
var tracker: synch.RegOut = synch.RegOut_static.getInstance(sidl_ex);
var magicNumber = 13;
//tracker.setExpectations(76, sidl_ex);
tracker.setExpectations(-1, sidl_ex);

proc init_part(name: string) 
{
  part_no += 1;
  tracker.startPart(part_no, sidl_ex);
  tracker.writeComment("Method " + name, sidl_ex);
  magicNumber = clearstack(magicNumber);
}


proc run_part(result: string, template: string)
{
  var r: ResultType;
  tracker.writeComment("should return " + template, sidl_ex);
  if (result == template) then
    r = ResultType.PASS;
  else 
    r = ResultType.FAIL;
  tracker.endPart(part_no, r, sidl_ex);
  tracker.writeComment("Method returned " + result, sidl_ex);
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
  tracker.writeComment("Class C:", sidl_ex);
  init_part("c.c()"); run_part(c.c(sidl_ex), "C.c");
}

  {
    var d = new Inherit.D();
    tracker.writeComment("Class D: inheritance of interface A", sidl_ex);
    init_part("d.a()"); run_part(d.a(sidl_ex), "D.a");
    init_part("d.d()"); run_part(d.d(sidl_ex), "D.d");

    tracker.writeComment("Class D: via interface A", sidl_ex);
    var a = new Inherit.A(d.cast_Inherit_A());
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting D to interface A", sidl_ex);
    if ( a == nil) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("a.a()"); run_part(a.a(sidl_ex), "D.a");
    }

    tracker.writeComment("Class D2: via interface A", sidl_ex);
    var d2 = new Inherit.D(cast_Inherit_D(a.ior));
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting A to interface D2", sidl_ex);
    if ( d2 == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("d2.d()"); run_part(d2.d(sidl_ex), "D.d");
    }

  }

  {
    var e = new Inherit.E();
    tracker.writeComment("Class E: inheritance of class C", sidl_ex);
    init_part("e.c()"); run_part(e.c(sidl_ex), "C.c");
    init_part("e.e()"); run_part(e.e(sidl_ex), "E.e");

    tracker.writeComment("Class E: via class C (C.c not overridden)", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting E to class C", sidl_ex);
    var c = new Inherit.C(e.cast_Inherit_C());
    if ( c == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("c.c()"); run_part(c.c(sidl_ex), "C.c");
    }
  }

  {
    var e2 = new Inherit.E2();
    tracker.writeComment("Class E2: inheritance of class C", sidl_ex);
    init_part("e2.c()"); run_part(e2.c(sidl_ex), "E2.c");
    init_part("e2.e()"); run_part(e2.e(sidl_ex), "E2.e");

    tracker.writeComment("Class E2: via class C (C.c overridden)", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting E2 to class C", sidl_ex);
    var c = new Inherit.C(e2.cast_Inherit_C());
    if ( c == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("c.c()"); run_part(c.c(sidl_ex), "E2.c");
    }

    init_part("Inherit::E2::m()"); run_part(Inherit.E2_static.m(sidl_ex), "E2.m");
  }

  {
    var f = new Inherit.F();
    tracker.writeComment("Class F: Multiple inheritance (no overriding)", sidl_ex);
    init_part("f.a()"); run_part(f.a(sidl_ex), "F.a");
    init_part("f.b()"); run_part(f.b(sidl_ex), "F.b");
    init_part("f.c()"); run_part(f.c(sidl_ex), "C.c");
    init_part("f.f()"); run_part(f.f(sidl_ex), "F.f");
    
    tracker.writeComment("Class F: via interface A", sidl_ex) ;
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F to class A", sidl_ex);
    var a = new Inherit.A(f.cast_Inherit_A());
    if ( a == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("a.a()"); run_part(a.a(sidl_ex), "F.a");
    }

    tracker.writeComment("Class F: via interface B", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F to interface B", sidl_ex);
    var b = new Inherit.B(f.cast_Inherit_B());
    if ( b == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("b.b()"); run_part(b.b(sidl_ex), "F.b");
    }


    tracker.writeComment("Class F: via class C (no overloading of C.c)", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F to class C", sidl_ex);
    var c = new Inherit.C(f.cast_Inherit_C());
    if ( c == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("c.c()"); run_part(c.c(sidl_ex), "C.c");
    }
  }

  {
    var f2 = new Inherit.F2();
    tracker.writeComment("Class F2: Multiple inheritance (overrides C.c)", sidl_ex);
    init_part("f2.a()"); run_part(f2.a(sidl_ex), "F2.a");
    init_part("f2.b()"); run_part(f2.b(sidl_ex), "F2.b");
    init_part("f2.c()"); run_part(f2.c(sidl_ex), "F2.c");
    init_part("f2.f()"); run_part(f2.f(sidl_ex), "F2.f");
    
    tracker.writeComment("Class F2: via interface A", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F2 to interface A", sidl_ex);
    var a = f2;
    if ( a == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("a.a()"); run_part(a.a(sidl_ex), "F2.a");
    }

    tracker.writeComment("Class F2: via interface B", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F2 to interface B", sidl_ex);
    var b = f2;
    if ( b == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("b.b()"); run_part(b.b(sidl_ex), "F2.b");
    }

    tracker.writeComment("Class F2: via class C (overloads C.c)", sidl_ex) ;
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting F2 to class C", sidl_ex);
    var c = f2;
    if ( c == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("c.c()"); run_part(c.c(sidl_ex), "F2.c");
    }
  }

  {
    var g = new Inherit.G();

    tracker.writeComment("Class G: indirect multiple inheritance ( no overloads)", sidl_ex);
    init_part("g.a()"); run_part(g.a(sidl_ex), "D.a");
    init_part("g.d()"); run_part(g.d(sidl_ex), "D.d");
    init_part("g.g()"); run_part(g.g(sidl_ex), "G.g");

    
    tracker.writeComment("Class G: via interface A", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting G to interface A", sidl_ex);
    var a = g;
    if ( a == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("a.a()"); run_part(a.a(sidl_ex), "D.a");
    }

    tracker.writeComment("Class G: via class D", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting G to class D", sidl_ex);
    var d = g;
    if ( d == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("d.a()"); run_part(d.a(sidl_ex), "D.a");
      init_part("d.d()"); run_part(d.d(sidl_ex), "D.d");
    }

  }
  
  {
    var g2 = new Inherit.G2();

    tracker.writeComment("Class G2: indirect multiple inheritance (overloads)", sidl_ex);
    init_part("g2.a()"); run_part(g2.a(sidl_ex), "G2.a");
    init_part("g2.d()"); run_part(g2.d(sidl_ex), "G2.d");
    init_part("g2.g()"); run_part(g2.g(sidl_ex), "G2.g");

    
    tracker.writeComment("Class G2: via interface A", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting G2 to interface A", sidl_ex);
    var a = g2;
    if ( a == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("a.a()"); run_part(a.a(sidl_ex), "G2.a");
    }

    tracker.writeComment("Class G2: via class D", sidl_ex);
    part_no += 1;
    tracker.startPart(part_no, sidl_ex);
    tracker.writeComment("Casting G2 to class D", sidl_ex);
    var d = g2;
    if ( d == nil ) {
      tracker.endPart(part_no, synch.ResultType.FAIL, sidl_ex);
    } else {
      tracker.endPart(part_no, synch.ResultType.PASS, sidl_ex);
      init_part("d.a()"); run_part(d.a(sidl_ex), "G2.a");
      init_part("d.d()"); run_part(d.d(sidl_ex), "G2.d");
    }

  }

  {
    var i = new Inherit.I();
    tracker.writeComment("Class I: implements abstract class H that implements A", sidl_ex);
    init_part("i.a()"); run_part(i.a(sidl_ex), "I.a");
    init_part("i.h()"); run_part(i.h(sidl_ex), "I.h");
    
    tracker.writeComment("Class I: via interface A", sidl_ex);
    var a = i;
    init_part("a.a()"); run_part(a.a(sidl_ex), "I.a");
    
    tracker.writeComment("Class I: via abstract class H", sidl_ex);
    var h = i;
    init_part("h.a()"); run_part(h.a(sidl_ex), "I.a");
    init_part("h.h()"); run_part(h.h(sidl_ex), "I.h");
  }

  {
    var j = new Inherit.J();
    tracker.writeComment("\nClass J: implements A and B, extends E. Calls super of E and C\n", sidl_ex);
    init_part("j.a()"); run_part(j.a(sidl_ex), "J.a");
    init_part("j.b()"); run_part(j.b(sidl_ex), "J.b");
    init_part("j.j()"); run_part(j.j(sidl_ex), "J.j");
    init_part("j.c()"); run_part(j.c(sidl_ex), "J.E2.c");
    init_part("j.e()"); run_part(j.e(sidl_ex), "J.E2.e");
    
    init_part("Inherit.J_static.m()"); run_part(Inherit.J_static.m(sidl_ex), "E2.m");
  }

  /* { */
  /*   var k = new Inherit.K(); */
  /*   tracker.writeComment("Class K: implements A2, extends H.", sidl_ex); */
  /*   init_part("k.a()"); run_part(k.a(sidl_ex), "K.a"); */
  /*   init_part("k.a(0)"); run_part(k.a(0), "K.a2"); */
  /*   init_part("k.h()"); run_part(k.h(sidl_ex), "K.h"); */
  /*   init_part("k.k()"); run_part(k.k(sidl_ex), "K.k"); */
    
  /*   tracker.writeComment("Class K: via interface A", sidl_ex); */
  /*   var a = k; */
  /*   init_part("a.a()"); run_part(a.a(sidl_ex), "K.a"); */
   
  /*   tracker.writeComment("Class K: via interface A2", sidl_ex); */
  /*   var a2 = k; */
  /*   init_part("a2.a(0)"); run_part(a2.a(0), "K.a2"); */
    
  /*   tracker.writeComment("Class K: via abstract class H", sidl_ex); */
  /*   var h = k; */
  /*   init_part("h.a()"); run_part(h.a(sidl_ex), "K.a"); */
  /*   init_part("h.h()"); run_part(h.h(sidl_ex), "K.h"); */
  /* } */

  /* { */
  /*   var l = new Inherit.L(); */
  /*   tracker.writeComment("Class L: implements A, A2.", sidl_ex); */
  /*   init_part("l.a()"); run_part(l.a(sidl_ex), "L.a"); */
  /*   init_part("l.a(0)"); run_part(l.a(0), "L.a2"); */
  /*   init_part("l.l()"); run_part(l.l(sidl_ex), "L.l"); */
    
  /*   tracker.writeComment("Class L: via interface A", sidl_ex); */
  /*   var a = l; */
  /*   init_part("a.a()"); run_part(a.a(sidl_ex), "L.a"); */
   
  /*   tracker.writeComment("Class L: via interface A2", sidl_ex); */
  /*   var a2 = l; */
  /*   init_part("a2.a(0)"); run_part(a2.a(0), "L.a2"); */
    
  /* } */

tracker.close(sidl_ex);

