ACLOCAL_AMFLAGS=-I m4

python_PYTHON= 
nobase_python_PYTHON= 
noinst_PYTHON= 
BUILT_SOURCES= 
nobase_pyexec_LTLIBRARIES=
bin_SCRIPTS=

EXTRA_DIST= \
	COPYRIGHT README \
	contrib/README \
	contrib/argparse-1.1/setup.py      \
	contrib/argparse-1.1/README.txt    \
	contrib/argparse-1.1/PKG-INFO      \
	contrib/argparse-1.1/NEWS.txt      \
	contrib/argparse-1.1/argparse.py   \
	contrib/doxypy-0.4.2/PKG-INFO  \
	contrib/doxypy-0.4.2/setup.py  \
        contrib/doxypy-0.4.2/doxypy.py \
	contrib/ply/README             \
	contrib/ply/__init__.py        \
	contrib/ply/cpp.py	       \
	contrib/ply/ctokens.py	       \
	contrib/ply/lex.py	       \
	contrib/ply/yacc.py

# Recursive make considered harmful
include braid/Makefile.include

DISTCHECK_CONFIGURE_FLAGS=--disable-documentation
if ENABLE_DOCUMENTATION
EXTRA_DIST+= \
	doc/html/index.html \
	doc/latex/*.pdf     \
	doc/html/*.*        \
	doc/html/search/*.*
braid/braid: $(BUILT_SOURCES) doc/html/index.html
endif

.PHONY: doc
doc:
	@echo "  DOXYGEN"
	@doxygen >/dev/null || echo "Doxygen not found; SKIPPING"
	$(MAKE) -j1 -C doc/latex >/dev/null
doc/html/index.html:
	$(MAKE) doc

info_TEXINFOS = doc/braid.texi

distclean-local:
	rm -f toolcheck.log

clean-local: clean-braid
	rm -rf doc/html
	rm -rf doc/latex
	rm -rf doc/man


if PYTHON_VERSION_2_6
# add Backports
  python_PYTHON += contrib/argparse-1.1/argparse.py
endif

