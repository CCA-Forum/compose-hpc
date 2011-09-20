# Note: If the ROSE header and library files are not in standard places,
# define ROSEINSTALL in your environment:

ifdef SEXPRINSTALL
CFLAGS += -I$(SEXPRINSTALL)
LDFLAGS += -L$(SEXPRINSTALL)
endif
ifdef ROSEINSTALL
CFLAGS += -I$(ROSEINSTALL)/include
LDFLAGS += -L$(ROSEINSTALL)/lib
endif
ifdef BOOSTINSTALL
CFLAGS += -I$(BOOSTINSTALL)/include
LDFLAGS += -L$(BOOSTINSTALL)/lib
endif

EXEC=annotation_test
CCFILES=KVAnnotation.cpp SXAnnotation.cpp ${EXEC}.cpp 
HFILES=KVAnnotation.h SXAnnotation.h Annotation.h
OFILES=$(patsubst %.cc,%.o,${CCFILES})
LIBS=-lrose
CXX=g++
SOURCES=${CCFILES}
OUTS=$(addprefix out/, $(basename $(notdir ${SOURCES})))
OUTS_O=$(addprefix out-o/, $(basename $(notdir ${SOURCES})))

.PHONY: main
main: ${EXEC}

${EXEC}: ${OFILES}
	${CXX} $^ ${CFLAGS} ${LDFLAGS} ${LIBS} -o $@

%.o: %.cc ${HFILES}
	${CXX} -Wall -O2 -ansi ${CFLAGS} -c $<

run: out

.PHONY: out
out: ${EXEC}
	rm -rf $@
	mkdir $@
	for f in ${EXAMPLES} ${TESTS} ; do b=$$(basename "$$f") ; echo ----- $$b ----- ; ./${EXEC} -rose:o out/$$b $$f ; done

out/%: out/%.c
	mpicc -o $@ $<

.PHONY: mpibuild
mpibuild: ${OUTS}

# Create skeletons via outlining, not removing

.PHONY: out-o
out-o: ${EXEC}
	rm -rf $@
	mkdir $@
	for f in ${EXAMPLES} ${TESTS} ; do b=$$(basename "$$f") ; echo ----- $$b ----- ; ./${EXEC} -rose:o out-o/$$b -local:outline $$f ; done

out-o/%: out-o/%.c
	mpicc -o $@ $<

.PHONY: mpibuild
mpibuild-o: ${OUTS_O}

  # ^ not fully successful, use "make -k mpibuild-o" for now.

.PHONY: compare
compare: ${OUTS}
	@for f in out/*.c ; do b=$$(basename "$$f") ; if [ -e expected/$$b.expected ] ; then diff $$f expected/$$b.expected ; fi ; done

.PHONY: test
test: ${EXEC}
	${MAKE} run
	${MAKE} mpibuild
	${MAKE} compare

TAGS: ${CCFILES} ${HFILES}
	etags $^

show:
	@echo $($(VALUE))              # note: this expands!
