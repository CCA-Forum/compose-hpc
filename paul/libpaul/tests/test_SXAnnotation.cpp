#include <iostream>
#include <stdio.h>

#include "rose.h"

#include <paul/SXAnnotationValue.h>

int main( int argc, char * argv[] )
{
  string s0 = "(a b c d)";

  SXAnnotationValue v (s0);
  v.print();
  cout << endl;

  Annotation a = Annotation(s0, (SgLocatedNode *)(NULL), "tag", &v);

  cout << "tag= " << a.getTag() << endl;
  cout << "str= " << a.getValueString() << endl;
  SXAnnotationValue *p = isSXAnnotationValue(a.getValue());
  p->print();

  cout << "Finished\n";
  return 0;
}
