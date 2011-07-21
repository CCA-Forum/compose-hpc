#ifndef COMMENTVISITOR_H
#define COMMENTVISITOR_H

class CommentVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
public:
  CommentVisitor(std::string, int*);
  std::string inpFile;
  int *fileCount;
};

#endif
