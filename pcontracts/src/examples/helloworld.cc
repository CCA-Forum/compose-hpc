/* NOTE:  (Class) invariants are NOT supported for non-instance routines. */

#include <iostream>

using namespace std;

const string gesture = "Hello";

/* %CONTRACT REQUIRE have_gesture: !gesture.empty(); have_name: !nm.empty(); */
void
printGreeting(string nm)
{
  cout << gesture << " " << nm << "!\n";
}

/* %CONTRACT INIT */
/* %CONTRACT FINAL */
int 
main(int argc, char*argv[]) {
  string names[] = {
    "David",
    "Tom",
    "",
    "world"
  };

  for (int i=0; i<4; i++) {
    printGreeting(names[i]);
  }

  return 0;
}
