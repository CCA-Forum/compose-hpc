/* 
 * CONTRACT INVARIANT annotations NOT supported for non-instance methods
 * so none are added to this example.
 */
#include <iostream>

using namespace std;

const string gestureStart = "Hello";
const string gestureEnd = "Good bye";

/* %CONTRACT REQUIRE have_gesture: !gesture.empty(); have_name: !nm.empty(); */
void
printGesture(const string gesture, string nm)
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

  /* %CONTRACT INIT helloworld-v3.config; */

  for (int i=0; i<4; i++) {
    printGesture(gestureStart, names[i]);
  }

  /*
   * Warning: A STATS here (for some TBD reason) results in skipping the
   * instrumentation of the FINAL below.  Debugging indicates no nodes 
   * after contents of the printGesture() line below are visited.
   */

  cout << "Sure is a beautiful day!\n";
  
  for (int i=0; i<4; i++) {
    printGesture(gestureEnd, names[i]);
  }

  /* %CONTRACT FINAL */

  cout << "Until we meet again.\n";

  return 0;
}
