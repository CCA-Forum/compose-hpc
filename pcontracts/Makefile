ROOT_DIR=.
include make_test.inc

SUBDIRS = src

all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) all) \
	done;

docs: cleandocs
	@ln -s $(ROSE_HOME) rose; \
	doxygen; \
	cp COPYRIGHT docs/html

check: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) check) \
	done; \
	$(MAKE) checksummary; \
	echo; echo "DONE (check)"; echo

checkall: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) checkall) \
	done; \
	$(MAKE) checksummary; \
	echo "DONE (checkall)"; echo

checksummary:
	@echo; \
	echo "  *************************************************************";\
	echo "  * Standard out messages must be reviewed manually.          *";\
	echo "  *************************************************************";\
	echo

clean: 
	@rm -f $(CHECK_STATUS); \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) clean) \
	done;

cleanall:  cleandocs
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) cleanall) \
	done; \
	rm -f *.out $(CHECK_STATUS)* ./bin/*

cleandocs:
	@rm -rf rose docs

help:
	@echo; \
	echo "  *************************************************************";\
	echo "  * Standard out messages must be reviewed manually.          *";\
	echo "  *                                                           *";\
	echo "  * NOTE:                                                     *";\
	echo "  * - 'make check' checks the core software.                  *";\
	echo "  * - 'make checkall' checks core AND instrumented codes.     *";\
	echo "  *                                                           *";\
	echo "  * Exercising instrumented codes requires the software be    *";\
	echo "  * built with 'PCEFLAGS=-DPAUL_CONTRACTS'.  It can be added  *";\
	echo "  * during check (ie, 'make PCEFLAGS=-DPAUL_CONTRACTS check') *";\
	echo "  * to perform the build, execution, and check processes.     *";\
	echo "  *************************************************************";\
	echo
