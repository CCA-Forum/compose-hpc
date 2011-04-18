#include <iostream>
#include "rose.h"
#include "CommentVisitor.h"

#define CPP_COMMENT (1)
#define C_COMMENT   (2)

using namespace std;

void CommentVisitor::visit(SgNode *node) {
  SgLocatedNode *locatedNode = isSgLocatedNode(node);
  if(locatedNode != NULL) {
    AttachedPreprocessingInfoType *preprocInfo;
    preprocInfo = locatedNode->getAttachedPreprocessingInfo();
    if(preprocInfo != NULL) {
      AttachedPreprocessingInfoType::iterator i;
      for(i=preprocInfo->begin();i != preprocInfo->end(); i++) {
        switch ((*i)->getTypeOfDirective()) {
          case C_COMMENT:
          case CPP_COMMENT:
            string comment = (*i)->getString();
            cout << node->class_name() << endl;
            cout << comment << endl;
            break;
        }
      }
    }
  }
}
