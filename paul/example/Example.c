#include <stdio.h>

/* This is an example of a C file that includes PAUL annotations. A PAUL
annotation is structured as follows (within a comment):

1. Any amount of whitespace
2. An annotation tag, some (user-specifiable, possibly with language-specific
   default) short string that tells PAUL to try to process the comment.
3. A PAUL annotation identifier (AID)
4. A single whitespace character (including newline)
5. An arbitrary string that will be passed to an annotation-specfic parser, 
   identified by the AID.  The string is terminated by the end of the comment, 
   and may include whitespace and newlines.

In addition, PAUL provides a default parser that processes a sequence of
"key=value" pairs. The keys must be the usual sort of identifiers, and the
values can be any of the following:

* Integers, e.g. 500
* Floats, e.g. 3.1
* Bools, e.g. True
* Quoted escaped strings (e.g. "Hello there\n") and here documents
* Alphanumeric identifiers

Other types? Note that a client can always override the default parser and
parse the annotation text (i.e. everything after the AID) themselves.

*/

/*%%% CONTRACT w="Multi-line\nstrings are fine" x=Identifier y=5.6
               z=True */


/*%%% CODE code=<<EOF
int bar(int x, int y) {
  printf("Test this thing out\n");
  return (x * y);
}
EOF x=5
*/

/*
               h=
a.begin(EOF)
b.begin(EOF.preamble)
b.end(EOF.preamble)
foo
b.begin(EOF.suffix)
a.end(EOF)
*/

/* FORTRAN90 block literals
!%%% CONTRACT blah=whatever h=
!begin(EOF)
!this is my block
!end(EOF)
*/

double foo(double x, double y) {
  
  int x = 6;
  /*%%% LOOP */
  for(int i=0; i < 10; i++) {
    x = x + i;
  }
  return x * y;
}

/*%%% RENAME name=oldmain */
int main() {
  printf("Hello world\n");
  return 0;
}
