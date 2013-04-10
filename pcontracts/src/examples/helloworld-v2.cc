/* 
 * CONTRACT INVARIANT annotations NOT supported for non-instance methods
 * so none are added to this example.
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

/* %CONTRACT INIT */

  for (int i=0; i<4; i++) {
    printGreeting(names[i]);
  }

/* %CONTRACT FINAL */

  return 0;
}
