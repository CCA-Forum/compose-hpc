#include <iostream>

using namespace std;

// This is a comment
//%TEST1 x="hello world" y = 5
int main() {
  int x = 0;
  
  /*%TEST2 */
  for(int i=0; i < 10; i++) {
    x += i;
  }
  cout << x << endl;
  return 0;
}
