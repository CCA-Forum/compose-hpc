#include <stdio.h>
#include <stdlib.h>

/** Basic void method with multiple return statements. */
void
checkValue(int i);

/** Range check for invariant clause. */
int
isValid();
int d_i;


/* %CONTRACT REQUIRE i > -6 && i < 500; */
void
checkValue(int i)
{
  switch (i)
  {
    case 1:
      printf("i is the first arbitrary value (%d)\n", i);
      return;
    case 7:
    case 11:
    case 13:
    case 25:
      printf("i (%d) is in the arbitrary set of values\n", i);
      return;
    default:
      printf("i (%d) is NOT one of the arbitrary values\n", i);
  }

  return;
}

/* %CONTRACT ENSURE is pure; */
int
isValid()
{
  return -500 <= d_i && d_i <= 500;
}


/** * Driver for simple test.  */
/* %CONTRACT INIT */
/* %CONTRACT FINAL */
int 
main(int argc, char **argv) {
  /* Checks should fail only if contracts enforced. */
  int values[7] = { -5, 0, 7, 10, 11, 500, 501 };

  for (unsigned int i=0; (i<7); i++)
  {
    d_i = values[i];
    checkValue(values[i]);
  }

  return 0;
} /* main */
