ACLOCAL_AMFLAGS = -I m4
# initialize everything to blank so the include files can use the += operator
python_PYTHON= 
nobase_python_PYTHON= 
noinst_PYTHON= 
nobase_include_HEADERS=
BUILT_SOURCES= 
nobase_pyexec_LTLIBRARIES=
bin_SCRIPTS=
TESTS=
TEST_EXTENSIONS=

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
include regression/interop/Makefile.include
include chapel-patches/Makefile.include

DISTCHECK_CONFIGURE_FLAGS=--disable-documentation \
  PKG_CONFIG_PATH=$(prefix)/lib/pkgconfig:$(PKG_CONFIG_PATH)

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
	@echo "  GEN [doxygen documentation]"
	($(DOXYGEN) >/dev/null && $(MAKE) MAKEFLAGS=-j1 -C doc/latex >/dev/null) \
        || echo "Doxygen not found; SKIPPING"

doc/html/index.html:
	$(MAKE) doc

info_TEXINFOS = doc/braid.texi

distclean-local:
	rm -f toolcheck.log

clean-local: clean-braid
	rm -rf doc/html
	rm -rf doc/latex
	rm -rf doc/man
	if [ "x$(builddir)" != "x$(srcdir)" ]; then \
		cd $(builddir) && \
                rm -rf $(BABEL_TESTS) \
                       regression/interop/output && \
		for f in `echo $(BABEL_TESTS)|sed 's/ /\\n/g'`; do \
	            rm -rf `dirname \`dirname $$f\``; \
		done; \
	fi


if PYTHON_VERSION_2_6
# add Backports
  python_PYTHON += contrib/argparse-1.1/argparse.py
endif

########################################################################
# Babel functionality: 

# Runtime support files
nobase_include_HEADERS += \
  chpl/chpl_sidl_array.h \
  chpl/sidl.chpl

BUILT_SOURCES += chpl/sidl.chpl
EXTRA_DIST += chpl/sidl.inc chpl/array.inc

noinst_SCRIPTS = patch-chapel-compiler.sh.in

chpl/sidl.chpl: chpl/sidl.inc chpl/array.inc 
	@mkdir -p chpl
	$(AM_V_GEN)
	@$(CPP) $(AM_CPPFLAGS) -I$(srcdir)/chpl -x c -P -C $< -o $@
