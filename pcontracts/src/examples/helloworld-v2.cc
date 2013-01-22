/* NOTE:  Invariants NOT supported for C at this time. */
#include <iostream>

using namespace std;

/* %CONTRACT INVARIANT have_gesture: !gesture.empty(); */

const string gesture = "Hello";

/* %CONTRACT REQUIRE !gesture.empty(); have_name: !nm.empty(); */
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

/* %CONTRACT INIT */

  for (int i=0; i<4; i++) {
    printGreeting(names[i]);
  }

/* %CONTRACT FINAL */

  return 0;
}
