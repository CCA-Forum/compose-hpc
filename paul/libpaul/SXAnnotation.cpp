#include "SXAnnotation.h"

SXAnnotation::SXAnnotation(string s, SgNode *n) : Annotation::Annotation(s, n) {
  std::cout << "SXAnnotation CTR: " << s << std::endl;
  // parse s here
}
