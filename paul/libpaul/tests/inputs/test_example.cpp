#include <iostream>

using namespace std;

// This is a comment
//%EXAMPLE x="hello world" y = 5
int main() {
  int x = 0;

  for(int i=0; i < 10; i++) {
    //%EXAMPLE blah=blah
    x += i;
  }
  cout << x << endl;
  return 0;
}

//%EXAMPLE x="hello world" y = 5 z = 9
int x = 5;

