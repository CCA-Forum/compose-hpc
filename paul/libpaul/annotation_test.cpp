#include "Annotation.h"
#include "KVAnnotation.h"
#include "SXAnnotation.h"

int main(int argc, char **argv) {
  string s = "I am a test.";

  KVAnnotation kva(s, null);
  SXAnnotation sxa(s, null);

}
