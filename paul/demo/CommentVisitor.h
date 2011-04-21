#ifndef COMMENTVISITOR_H
#define COMMENTVISITOR_H

class CommentVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
};

#endif
