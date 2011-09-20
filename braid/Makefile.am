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
	contrib/argparse-1.1/LICENSE       \
	contrib/argparse-1.1/NEWS.txt      \
	contrib/argparse-1.1/PKG-INFO      \
	contrib/argparse-1.1/README.txt    \
	contrib/argparse-1.1/argparse.py   \
	contrib/argparse-1.1/setup.py      \
	contrib/doxypy-0.4.2/LICENSE       \
	contrib/doxypy-0.4.2/PKG-INFO      \
        contrib/doxypy-0.4.2/doxypy.py     \
	contrib/doxypy-0.4.2/setup.py      \
	contrib/ply/README                 \
	contrib/ply/__init__.py            \
	contrib/ply/cpp.py	           \
	contrib/ply/ctokens.py	           \
	contrib/ply/lex.py	           \
	contrib/ply/yacc.py

# Recursive make considered harmful
include braid/Makefile.include
include regression/interop/Makefile.include
include chpl/Makefile.include
include chapel-patches/Makefile.include

DISTCHECK_CONFIGURE_FLAGS=--disable-documentation \
  PKG_CONFIG_PATH=$(prefix)/lib/pkgconfig:$(PKG_CONFIG_PATH)

if ENABLE_DOCUMENTATION
EXTRA_DIST+= \
	doc/html/index.html \
	doc/html/*.*        \
	doc/html/search/*.*
#	doc/latex/*.pdf     # I say this is just useless bloat

braid/braid: $(BUILT_SOURCES) doc/html/index.html
endif

.PHONY: doc
doc:
	@echo "  GEN [doxygen documentation]"
	@($(DOXYGEN) >/dev/null && $(MAKE) MAKEFLAGS=-j1 -C doc/latex >/dev/null) \
        || echo "Doxygen not found; SKIPPING"

doc/html/index.html:
	$(MAKE) doc

info_TEXINFOS = doc/braid.texi
dist_man_MANS = $(srcdir)/doc/braid.1

# We are distributing the compiled manpage so we don't depend on rst2man
# Insert the output of braid --help into the MAN page.
EXTRA_DIST += doc/man_header.txt
$(srcdir)/doc/braid.1: doc/man_header.txt $(top_builddir)/braid/braid COPYRIGHT
	$(AM_V_GEN)
	@cat $< |sed -e 's/@DATE@/$(shell date +%Y-%m-%d)/g' \
                     -e 's/\@VERSION\@/$(PACKAGE_VERSION)/g' >braid.txt
	@sh $(top_builddir)/braid/braid --help \
	|  sed -e 's/usage:/\nSYNOPSIS\n========\n\n/' \
               -e 's/:$$/\n~~~~~~~~~~~~~~~~~~~~\n\n/' \
               -e 's/^Braid/\nDESCRIPTION\n===========\n\nBraid/' \
               -e 's/pos/OPTIONS\n=======\n\npos/' >>braid.txt
	@echo >>braid.txt
	@cat $(srcdir)/COPYRIGHT >>braid.txt
	@if test -w $@; then \
             if which rst2man >/dev/null; then \
                 rst2man -v braid.txt $@; \
             else echo "**WARNING rst2man not installed, did not update $@"; \
             fi \
         else echo "**WARNING $@ is not writable, did not update. (This is OK during distcheck)"; \
         fi
	@rm -f braid.txt

CLEANFILES = toolcheck.log

clean-local: clean-braid
	rm -rf doc/html
	rm -rf doc/latex
	rm -rf doc/man
	rm -rf chpl/sidl_*.[ch] chpl/sidl.h
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
  chpl/sidl.chpl \
  $(SIDLSTUB_HDRS)

CHPL_RUNTIME_FILES = chpl/array.inc chpl/barray.inc chpl/array_support.inc
BUILT_SOURCES += chpl/sidl.chpl $(SIDLSTUB_SRCS)
EXTRA_DIST += $(CHPL_RUNTIME_FILES) chpl/darray.inc

noinst_SCRIPTS = patch-chapel-compiler.sh.in

chpl/sidl.chpl: $(CHPL_RUNTIME_FILES) $(nobase_python_PYTHON)
	@$(MKDIR_P) chpl
	$(AM_V_GEN)
	@echo "generating Chapel bindings for the SIDL runtime"
	@cd chpl && \
          env PYTHONPATH=../braid:../braid/.libs:../$(top_srcdir)/contrib/argparse-1.1:../$(top_srcdir)/contrib/ply:$(PYTHONPATH) $(PYTHON) ../$(top_srcdir)/braid/braid.py \
            -cchpl $(SIDL_PATH)/sidl.sidl
	@echo "generating Chapel array extensions"
	@$(CPP) $(AM_CPPFLAGS) -I$(srcdir)/chpl -x c -P -C $(srcdir)/chpl/array.inc >>$@
	@$(CPP) $(AM_CPPFLAGS) -I$(srcdir)/chpl -x c -P -C $(srcdir)/chpl/barray.inc >>$@
	@$(CPP) $(AM_CPPFLAGS) -I$(srcdir)/chpl -x c -P -C $(srcdir)/chpl/array_support.inc >>$@

# Rules to build the Chapel stubs for the SIDL runtime
$(SIDLSTUB_SRCS): chpl/sidl.chpl
lib_LTLIBRARIES = libsidlstub_chpl.la
libsidlstub_chpl_la_SOURCES = $(SIDLSTUB_SRCS)
AM_CPPFLAGS = -I$(builddir)/chpl -I$(srcdir)/chpl -I$(BABEL_INCLUDEDIR) -I$(CHAPEL_ROOT)/runtime/include