class TestVisitor : public AstSimpleProcessing {
protected:
  void virtual visit(SgNode *node);
};
