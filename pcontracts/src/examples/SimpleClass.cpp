#include <iostream>
#include <string.h>
#include <stdlib.h>

using namespace std;

/* %CONTRACT INVARIANT isValid(); aNoop(); */ 

class SimpleClass
{
  private:
    int d_i;
    int d_count;

  public:
    /** Default constructor. */
    SimpleClass() : d_i(0), d_count(0) {}

    /** Simple constructor. */
    SimpleClass(int i) { d_i = i; d_count = 0; }

    /** Destructor.  Deletes the instance. */
    virtual ~SimpleClass() {}

    /** Basic method to selectively affect an attribute value. */
    void checkValue(int i);

    /** 
     * Ensure attribute value in the 'supported' range and return the new 
     * value.
     */
    int ensureValueInRange();

    /** Basic getter method to return current count. */
    int getCount();

    /** Basic method to modify attribute value. */
    void incrCount(int i);

    /** Range check for invariant clause. */
    bool isValid();

    /** A noop method to demontsrate multiple expressions in the invariant. */
    bool aNoop();

}; /* SimpleClass */



/** 
 * Basic method to selectively affect an attribute value.
 *
 * Purpose of this method is to have an example where:
 * 1) There are multiple empty return statements; and
 * 2) Attribute values may be modified.
 *
 * So invariant checks ONLY should be added prior to each return statement
 * and only when the visit instrumenter is used.
 */
void 
SimpleClass::checkValue(int i)
{
  switch (i)
  { 
    case 1: 
      cout << "i is the first arbitrary value (" << i << ")\n";
      incrCount(i);
      return;
    case 7: 
    case 11: 
    case 13: 
    case 25: 
      cout << "i (" << i << ") is in the arbitrary set of values\n";
      incrCount(i);
      return;
    default:
      cout << "i (" << i << ") is NOT one of the arbitrary values\n";
  } 

  return;
}


/** 
 * Basic getter method to return current count.
 *
 * Purpose of this method is to have an example where:
 * 1) there is a postcondition; and
 * 2) no attributes are changed.
 * 
 * So only postcondition checks should be added prior to the return.
 */
/* %CONTRACT ENSURE is pure; pce_result >= 0; */
int 
SimpleClass::getCount()
{
  return d_count;
}


/** 
 * Ensure attribute value in the 'supported' range and return the new value.
 *
 * Purpose of this method is to have an example: 
 * 1) with multiple return statements (including an explicit return of 0); 
 * 2) multiple postcondition expressions;
 * 3) invariant checks added; and
 * 4) a case where the postcondition will be violated.
 *
 * So postcondition and (if visit instrumenter used) invariant checks should
 * be added prior to each return.
 */
/* %CONTRACT ENSURE 
     pce_result >= 0; 
     pce_result <= 777; 
 */
int
SimpleClass::ensureValueInRange()
{
    if (d_i == 711) {
        d_i = 1024;
        return d_i;
    } else if ( (0 <= d_i) && (d_i <= 777) ) {
        return d_i;
    } else if (d_i < 0) {
        d_i = 0;
        return 0;
    } else {  // d_i > 777
        d_i = 777;
        return d_i;
    }
}


/** 
 * Basic method to modify attribute value.  
 *
 * Purpose of this method is to have an example: 
 * 1) with no explicit return;
 * 2) precondition and postcondition expressions;
 * 3) results in changing attribute values; and
 * 4) assertion expression with an embedded return.
 *
 * So precondition, postcondition, and (if visit instrumenter used) invariant
 * checks should be added to this method.
 */
/* %CONTRACT REQUIRE pce_in_range(i, 0, 
 1000000); */
/* %CONTRACT ENSURE getCount() > 0; */
void
SimpleClass::incrCount(int i) 
{
  d_count += i;
}


/** 
 * Range check for invariant clause. 
 *
 * No invariant checks should be added to this method since it 'is pure'.
 */
/* %CONTRACT ENSURE is pure; */
bool 
SimpleClass::isValid()
{
  return -1000000 <= d_i && d_i <= 1000000;
}


/** 
 * A noop method to demontsrate multiple expressions in the invariant.
 *
 * No invariant checks should be added to this method since it 'is pure'.
 */
/* %CONTRACT ENSURE is pure; */
bool 
SimpleClass::aNoop()
{
  return true;
}


/** Driver for SimpleClass test.  */
/* %CONTRACT INIT */
/* %CONTRACT FINAL */
int 
main(int argc, char **argv) {
  /* Checks should fail only if contracts enforced. */
  int values[6] = { -7, 0, 7, 11, 500, 711, };

  for (unsigned int i=0; (i<6); i++) 
  {
    SimpleClass* sc = new SimpleClass(values[i]);

    if (sc != NULL) {
      sc->checkValue(values[i]);
      int resValue = sc->ensureValueInRange();
      delete sc;
    }
  }

  return 0;
} /* main */
