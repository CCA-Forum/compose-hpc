BABELPATH=/home/prantl1/work/babel/install/bin
PYTHONPATH=../../../../braid.build/braid:../../braid:../../contrib/argparse-1.1

ifeq ($(SUM),)
NUMBERS=$(shell seq 1 24)
NUMBER_TYPES=$(patsubst %, %_bool, $(NUMBERS)) \
             $(patsubst %, %_string, $(NUMBERS))
#             $(patsubst %, %_float, $(NUMBERS)) 
#             $(patsubst %, %_int, $(NUMBERS)) 

NUMBER_TYPE_EXPRS=$(patsubst %, %_copy, $(NUMBER_TYPES))
# uncomment this line to run only specific tests
#NUMBER_TYPE_EXPRS=2_int_bsort 128_int_bsort #128_string_reverse 128_bool_nop 128_float_reverse
SIDLS=$(patsubst %,out/chpl_%.sidl, $(NUMBER_TYPE_EXPRS))
COMPILES=$(patsubst %,out/client_%/token, $(NUMBER_TYPE_EXPRS))
TIMINGS=$(patsubst %,out/client_%/times, $(NUMBER_TYPE_EXPRS))

TYPES=bool string #int float 
RESULTS=$(patsubst %, out/result_%_copy.txt, $(TYPES)) 

else

NUMBERS=1 $(shell seq 2 2 56)
NUMBER_TYPES=$(patsubst %, %_float, $(NUMBERS)) \
             #$(patsubst %, %_int, $(NUMBERS))

NUMBER_TYPE_EXPRS=$(patsubst %, %_sum, $(NUMBER_TYPES))
# uncomment this line to run only specific tests
#NUMBER_TYPE_EXPRS=2_int_bsort 128_int_bsort #128_string_reverse 128_bool_nop 128_float_reverse
SIDLS=$(patsubst %,out/chpl_%.sidl, $(NUMBER_TYPE_EXPRS))
COMPILES=$(patsubst %,out/client_%/token, $(NUMBER_TYPE_EXPRS))
TIMINGS=$(patsubst %,out/client_%/times, $(NUMBER_TYPE_EXPRS))

TYPES=float #int
RESULTS=$(patsubst %, out/result_%_sum.txt, $(TYPES)) 

endif
.PHONY: clean all sidls compile timings

all: graph.pdf

graph.pdf: graph.tex timings
	make $(RESULTS)
	env TEXINPUTS=~/sw/share/texmf//:$(TEXINPUTS) pdflatex -halt-on-error graph

out/result_%.txt:
	echo "n" "C" "CXX" "F77" "F90" "F03" "Java" "Python" >$@
	cat out/client_*_$(patsubst out/result_%.txt,%,$@)/times |sort -g >>$@

sidls: $(SIDLS)

compile: sidls 
	$(MAKE) $(COMPILES)
timings: compile
	$(MAKE) -j1 $(TIMINGS) # execute timing measurements serially

clean:
	rm -rf out graph.pdf

# this sed orgy seperates the parameters out of the filename:
#   out/chpl_1_bool_nop.sidl -> 1 bool nop
out/chpl_%.sidl:
	env PATH=$(BABELPATH):$(PATH) \
	    PYTHONPATH=$(PYTHONPATH) \
	    python benchgen2.py \
		$(shell echo $(patsubst out/chpl_%.sidl,%,$@) | sed 's/_.*//g') \
		$(shell echo $(patsubst out/chpl_%.sidl,%,$@) | sed 's/.*_\(.*\)_.*/\1/') \
		$(shell echo $(patsubst out/chpl_%.sidl,%,$@) | sed 's/.*._//g') 

out/client_%/times:
	@cd $(patsubst %/times,%,$@) && \
	 env PATH=$(BABELPATH):$(PATH) \
	    PYTHONPATH=$(PYTHONPATH) \
	    bash runAll.sh

out/C_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/CXX_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/F77_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/F90_%/token: out/chpl_%.sidl

	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/F03_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/Python_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/Java_%/token: out/chpl_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) CC=gcc -C $(patsubst %/token,%,$@)  all
	touch $@
out/client_%/token: out/chpl_%.sidl out/C_%/token out/CXX_%/token out/F77_%/token out/F90_%/token out/F03_%/token out/Python_%/token out/Java_%/token
	 #env PATH=$(BABELPATH):$(PATH) 
	TESTCASE=
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../C_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="" \
	        OUTFILE="runChapel2C"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../CXX_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="" \
	        OUTFILE="runChapel2CXX"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../F77_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="-lgfortran" \
	        OUTFILE="runChapel2F77"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../F90_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="-lgfortran" \
	        OUTFILE="runChapel2F90"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../F03_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="-lgfortran" \
	        OUTFILE="runChapel2F03"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../Python_%/libimpl*.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="" \
	        OUTFILE="runChapel2Python"
	$(MAKE) -C $(patsubst %/token,%,$@) \
		SERVER="$(patsubst out/client_%/token,../Java_%/libimpl.la,$@)" \
	        IMPL="main" \
		EXTRA_LDFLAGS="" \
	        OUTFILE="runChapel2Java"
	touch $@
