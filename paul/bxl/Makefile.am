python_PYTHON = bxl.py
bin_SCRIPTS   = bxl bxl-f
EXTRA_DIST    = README arith.bxl  aterm2contract.bxl  contract2aterm.bxl  demo.sh
dist_man_MANS = bxl.1
$(srcdir)/bxl.1: bxl.py
	help2man -n "minimalistic AWK work-a-like for context-free grammars" -s 1 -o $@ bxl
