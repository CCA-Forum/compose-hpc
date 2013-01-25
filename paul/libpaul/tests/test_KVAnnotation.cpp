#include <iostream>
#include <stdio.h>

#include "rose.h"

#include <paul/KVAnnotationValue.h>

int main( int argc, char * argv[] )
{
  string s0 = "k1 = 1 k2 = \"abc def\" k3 = 9000\n  ";
  string s1 = "k1 = 1 k3";

  KVAnnotationValue kva (s0);
  kva.print();
  cout << endl;
  cout << "k1: " << kva.lookup("k1")->string_value() << endl;
  cout << "k2: " << kva.lookup("k2")->string_value() << endl;
  cout << "k3: " << kva.lookup("k3")->string_value() << endl;

  Annotation a = Annotation(s0, (SgLocatedNode *)(NULL), "tag", &kva);

  cout << "tag= " << a.getTag() << endl;
  cout << "str= " << a.getValueString() << endl;
  KVAnnotationValue *p = isKVAnnotationValue(a.getValue());
  p->print();

  cout << "Finished\n";
  return 0;
}

