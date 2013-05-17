/* 
 * WARNING: CONTRACT INVARIANT annotations NOT supported for non-instance 
 * methods so none are added to the instrumented version of this example.
 */
#include <iostream>

using namespace std;

const string gesture = "Hello";

/* %CONTRACT REQUIRE have_gesture: !gesture.empty(); have_name: !nm.empty(); */
void
printGreeting(string nm)
{
  cout << gesture << " " << nm << "!\n";
}

int 
main(int argc, char*argv[]) {
  string names[] = {
    "David",
    "Tom",
    "",
    "world"
  };

/* %CONTRACT INIT helloworld.config; */

  for (int i=0; i<4; i++) {
    printGreeting(names[i]);
  }

/* 
 * WARNING: ROSE (17286) does NOT pass the final return AST node to the
 * visit routine so the visitor instrumenting it will NOT add the 
 * finalization routine below.  The current proposed work-around is
 * to add the annotation as a comment before the method definition.
 */
/* %CONTRACT FINAL */

  return 0;
}
