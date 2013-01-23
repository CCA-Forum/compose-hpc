python_PYTHON = bxl.py
bin_SCRIPTS   = bxl bxl-f
dist_man_MANS = bxl.1
$(srcdir)/bxl.1: bxl.py
	help2man -n "minimalistic AWK work-a-like for context-free grammars" -s 1 -o $@ bxl

EXTRA_DIST = README \
  examples/arith.bxl \
  examples/aterm2contract.bxl \
  examples/contract2aterm.bxl \
  examples/demo.sh

check-local:
	cd $(srcdir)/examples && sh demo.sh
