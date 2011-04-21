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
* Quoted escaped strings (e.g. "Hello there\n") and "here documents"
* Alphanumeric identifiers

Other types? Note that a client can always override the default parser and
parse the annotation text (i.e. everything after the AID) themselves.

*/

/*%%% CONTRACT w="Multi-line\nstrings are fine" x=Identifier y=5.6
               z=True */

// example of a heredoc:
/*%%% CODE code=<<EOF
int bar(int x, int y) {
  printf("Test this thing out\n");
  return (x * y);
}
EOF x=5
*/


//%%% GUARD cond=NonZero arg=x onErrorCall=failWith errorCallArg="x = 0"
double foo(double x, double y) {
  return y / x;
}

double foo(double x, double y) {
  if(x != 0) {
    failWith("x = 0");
  }
  return y / x;
}


/*%%% RENAME name=oldmain */
int main() {
  printf("Hello world\n");
  return 0;
}
