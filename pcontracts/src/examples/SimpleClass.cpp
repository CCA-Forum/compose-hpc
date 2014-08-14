#include <iostream>
#include <string.h>
#include <stdlib.h>

using namespace std;

/* %CONTRACT INVARIANT inRange(); */ 

class SimpleClass
{
  private:
    int d_i;

  public:
    /** Default constructor. */
    SimpleClass() : d_i(0) {}

    /** Simple constructor. */
    SimpleClass(int i) { d_i = i; }

    /** Destructor.  Deletes the instance. */
    virtual ~SimpleClass() {}

    /** Basic void method with multiple return statements. */
    void checkValue(int i);

    /** Range check for invariant clause. */
    bool isValid();

}; /* SimpleClass */

/** Basic void method with multiple return statements. */
/* %CONTRACT REQUIRE i > -6 && i <= 500; */
void 
SimpleClass::checkValue(int i)
{
  switch (i)
  { 
    case 1: 
      cout << "i is the first arbitrary value (" << i << ")\n";
      return;
    case 7: 
    case 11: 
    case 13: 
    case 25: 
      cout << "i (" << i << ") is in the arbitrary set of values\n";
      return;
    default:
      cout << "i (" << i << ") is NOT one of the arbitrary values\n";
  } 

  return;
}


/** Range check for invariant clause. */
/* %CONTRACT ENSURE is pure; */
bool 
SimpleClass::isValid()
{
  return -500 <= d_i && d_i <= 500;
}

/** Driver for SimpleClass test.  */
/* %CONTRACT INIT */
/* %CONTRACT FINAL */
int 
main(int argc, char **argv) {
  /* Checks should fail only if contracts enforced. */
  int values[7] = { -5, 0, 7, 10, 11, 500, 501 };

  for (unsigned int i=0; (i<7); i++) 
  {
    SimpleClass* sc = new SimpleClass(values[i]);

    if (sc != NULL) {
      sc->checkValue(values[i]);
      delete sc;
    }
  }

  return 0;
} /* main */
