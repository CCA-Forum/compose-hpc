class CommentVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
};
