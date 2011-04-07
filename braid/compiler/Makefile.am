if PYTHON_VERSION_2_6
# add Backports
  PYTHONPATH1=$(top_srcdir)/contrib:$(top_srcdir)/contrib/argparse-1.1
else
  PYTHONPATH1=$(top_srcdir)/contrib
endif

PYTHONPATH=$(builddir):$(PYTHONPATH1)

# list all Python source files that also contain unit tests
PYTHON_SOURCES = patmat.py sidl_oop.py codegen.py ir_handwritten.py braid.py splicer.py
python_PYTHON = $(PYTHON_SOURCES) sidl_parser.py
noinst_PYTHON = sidl_token.py
BUILT_SOURCES = parsetab.py ir.py sidl.py
GRAMMARS = ir.def sidl_or.def

bin_PROGRAMS = braid
braid_SOURCES =

pyexec_LTLIBRARIES = scanner.la
scanner_la_SOURCES = scanner.l
scanner_la_CPPFLAGS = $(PYTHONINC)
scanner_la_LDFLAGS = -avoid-version -module
scanner_la_LIBADD = -lfl

SIDL_TESTS= \
	../regression/parser/args.sidl \
	../regression/parser/ArrayTest.sidl \
	../regression/parser/cca-0.6.3.sidl \
	../regression/parser/cca-0.8.2.sidl \
	../regression/parser/cca.sidl \
	../regression/parser/clife.sidl \
	../regression/parser/construct.sidl \
	../regression/parser/cxxlife.sidl \
	../regression/parser/decaf.sidl \
	../regression/parser/enums.sidl \
	../regression/parser/ExceptionTest.sidl \
	../regression/parser/f90life.sidl \
	../regression/parser/hello-client.sidl \
	../regression/parser/hello-server.sidl \
	../regression/parser/hello.sidl \
	../regression/parser/hello2.sidl \
	../regression/parser/hooks.sidl \
	../regression/parser/inherit.sidl \
	../regression/parser/knapsack.sidl \
	../regression/parser/life.sidl \
	../regression/parser/objarg.sidl \
	../regression/parser/ordering.sidl \
	../regression/parser/output.sidl \
	../regression/parser/Overload.sidl \
	../regression/parser/pythonlife.sidl \
	../regression/parser/sidl.sidl \
	../regression/parser/sidlx.sidl \
	../regression/parser/sort.sidl \
	../regression/parser/strings.sidl \
	../regression/parser/strop.sidl \
	../regression/parser/struct.sidl \
	../regression/parser/vect.sidl \
	../regression/parser/wave2d_cxx_internal.sidl \
	../regression/parser/wave2d_cxx.sidl \
	../regression/parser/wave2d_f90.sidl \
	../regression/parser/wave2d.sidl \
	../regression/parser/wrapper.sidl

EXTRA_DIST=$(SIDL_TESTS) $(GRAMMARS) gramola.pl

## SIDL_TESTS=$(patsubst %.sidl,../regression/parser/%.sidl, $(SIDL_NAMES))

#UNIT_TOKENS = $(patsubst %.py,%.unit, $(python_PYTHON))
# Most of the Python files contain doctest tests
# see
# http://www.gnu.org/software/automake/manual/automake.html#Simple-Tests-using-parallel_002dtests
TESTS = $(PYTHON_SOURCES) $(SIDL_TESTS)
TEST_EXTENSIONS = .py .sidl
PY_LOG_COMPILER = env PYTHONPATH=$(PYTHONPATH) $(PYTHON)
AM_PY_LOG_FLAGS = -m doctest
SIDL_LOG_COMPILER = ./braid
AM_SIDL_LOG_FLAGS = --gen-sexp --gen-sidl


$(builddir)/%.unit: $(srcdir)/%.py scanner.la
	env PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m doctest $<

$(builddir)/%.success: $(srcdir)/%.sidl braid
	./braid $<

# Generate the parser tables
braid: braid.py ir.py $(PYTHON_SOURCES)
	@echo "LD_LIBRARY_PATH=$(PYTHON_SHARED_LIBRARY_DIR):LD_LIBRARY_PATH" >$@
	@echo "env PYTHONPATH=$(PYTHONPATH) $(PYTHON) -O $< \$$*" >>$@
	@chmod u+x $@

parsetab.py: sidl_ir.py sidl_parser.py scanner.la sidl_token.py
	@env PYTHONPATH=$(PYTHONPATH) $(PYTHON) $< --compile
	@env PYTHONPATH=$(PYTHONPATH) $(PYTHON) $< --compile

ir.py: $(srcdir)/ir.def $(srcdir)/gramola.pl
sidl.py: $(srcdir)/sidl.def $(srcdir)/gramola.pl

%.py: $(srcdir)/%.def
	@if [ -x $(SWIPL) ]; then \
	  egrep '^%[^%]' $< | sed 's/^%//g' >$@; \
	  $(SWIPL) -f $(srcdir)/gramola.pl -t main -q -s <$< >>$@; \
	else \
	  touch $@; \
	  echo "WARNING: $@ was NOT updated because swipl is missing"; \
	fi

clean-local:
	rm -rf scanner.c braid *.pyc *.pyo parsetab.py* lextab.py* parser.log parser.out
	rm -rf $(SIDL_TOKENS) $(UNIT_TOKENS)
# This can be done cleaner... Problem is, we do not want to force the
# dependency on swipl on the users
	if [ -x $(SWIPL) ]; then rm -f $(BUILT_SOURCES); fi
