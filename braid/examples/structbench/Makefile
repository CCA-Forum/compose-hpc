BABELPATH=/home/prantl1/work/babel/install/bin
PYTHONPATH=../../../../braid.build/braid:../../braid:../../../../install/lib/python2.6/site-packages:../../../../install/lib/python2.6/site-packages/braid

NUMBERS=96 128 1 2 4 6 8 12 16 24 32 48 64 #256 512
#NUMBERS=$(shell seq 1 16) 18 20 $(shell seq 24 4 48) 64 96 128
NUMBER_TYPES=$(patsubst %, %_float, $(NUMBERS)) \
             $(patsubst %, %_bool, $(NUMBERS)) \
             $(patsubst %, %_string, $(NUMBERS))

NUMBER_TYPE_EXPRS=$(patsubst %, %_reverse, $(NUMBER_TYPES)) \
                  $(patsubst %, %_nop,  $(NUMBER_TYPES)) \
                  $(patsubst %, %_int_bsort, $(NUMBERS))

#NUMBER_TYPE_EXPRS=2_int_bsort 128_int_bsort #128_string_reverse 128_bool_nop 128_float_reverse
SIDLS=$(patsubst %,out/struct_%.sidl, $(NUMBER_TYPE_EXPRS))
COMPILES=$(patsubst %,out/client_%/token, $(NUMBER_TYPE_EXPRS))
TIMINGS=$(patsubst %,out/client_%/times, $(NUMBER_TYPE_EXPRS))

TYPES=float bool string
RESULTS=$(patsubst %, out/result_%_reverse.txt, $(TYPES)) \
        $(patsubst %, out/result_%_nop.txt, $(TYPES)) \
        out/result_int_bsort.txt

.PHONY: clean all sidls compile timings

all: graph.pdf

graph.pdf: graph.tex timings
	make $(RESULTS)
	env TEXINPUTS=~/sw/share/texmf//:$(TEXINPUTS) pdflatex -halt-on-error graph

out/result_%.txt:
	echo "n C CXX F77 F90 F03 Java Python Chapel" >$@
	cat out/client_*_$(patsubst out/result_%.txt,%,$@)/times |sort -g >>$@

sidls: $(SIDLS)

compile: sidls 
	$(MAKE) $(COMPILES)
timings: compile
	$(MAKE) $(TIMINGS) # execute timing measurements serially

clean:
	rm -rf out graph.pdf

# this sed orgy seperates the parameters out of the filename:
#   out/struct_1_bool_nop.sidl -> 1 bool nop
out/struct_%.sidl:
	env PATH=$(BABELPATH):$(PATH) \
	    PYTHONPATH=$(PYTHONPATH) \
	    python benchgen.py \
		$(shell echo $(patsubst out/struct_%.sidl,%,$@) | sed 's/_.*//g') \
		$(shell echo $(patsubst out/struct_%.sidl,%,$@) | sed 's/.*_\(.*\)_.*/\1/') \
		$(shell echo $(patsubst out/struct_%.sidl,%,$@) | sed 's/.*._//g') 

out/client_%/times: out/client_%/outC out/client_%/outCXX out/client_%/outF77 out/client_%/outF90 out/client_%/outF03 out/client_%/outJava out/client_%/outPython out/client_%/outChapel
	@cd $(shell dirname $@) && sh combine.sh

out/C_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/Chapel_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all
	touch $@
out/CXX_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/F77_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/F90_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/F03_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/Python_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/Java_%/token: out/struct_%.sidl
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@
out/client_%/token: out/struct_%.sidl out/C_%/token out/Chapel_%/token out/CXX_%/token out/F77_%/token out/F90_%/token out/F03_%/token out/Python_%/token out/Java_%/token
	env PATH=$(BABELPATH):$(PATH) \
	$(MAKE) -C $(patsubst %/token,%,$@)  all >/dev/null
	touch $@

out/client_%/outC: out/client_%/token
	@cd $(shell dirname $@) && sh runC2C.sh
out/client_%/outCXX: out/client_%/token
	@cd $(shell dirname $@) && sh runC2CXX.sh
out/client_%/outF77: out/client_%/token
	@cd $(shell dirname $@) && sh runC2F77.sh
out/client_%/outF90: out/client_%/token
	@cd $(shell dirname $@) && sh runC2F90.sh
out/client_%/outF03: out/client_%/token
	@cd $(shell dirname $@) && sh runC2F03.sh
out/client_%/outJava: out/client_%/token
	@cd $(shell dirname $@) && sh runC2Java.sh
out/client_%/outPython: out/client_%/token
	@cd $(shell dirname $@) && sh runC2Python.sh
out/client_%/outChapel: out/client_%/token
	@cd $(shell dirname $@) && sh runC2Chapel.sh
