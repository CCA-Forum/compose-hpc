SUBDIRS = src
include make.inc

all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) all) \
	done;

docs: clean-docs
	@ln -s ${ROSE_HOME} rose; \
	doxygen; \
	cp COPYRIGHT docs/html

check: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) check) \
	done; \
	$(MAKE) checkSummary; \
	echo; echo "DONE (check)"; echo

checkAll: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) checkAll) \
	done; \
	$(MAKE) checkSummary; \
	echo "DONE (checkAll)"; echo

checkSummary:
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

clean-all:  cleanAll

cleanAll:  clean-docs
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) cleanAll) \
	done; \
	rm -f *.out $(CHECK_STATUS)*

clean-docs:
	@rm -rf rose docs

help:
	@echo; \
	echo "  *************************************************************";\
	echo "  * Standard out messages must be reviewed manually.          *";\
	echo "  *                                                           *";\
	echo "  * NOTE:                                                     *";\
	echo "  * - 'make check' checks the core software.                  *";\
	echo "  * - 'make checkAll' checks core AND instrumented codes.     *";\
	echo "  *                                                           *";\
	echo "  * Exercising instrumented codes requires the software be    *";\
	echo "  * built with 'PCEFLAGS=-DPAUL_CONTRACTS'.  It can be added  *";\
	echo "  * during check (ie, 'make PCEFLAGS=-DPAUL_CONTRACTS check') *";\
	echo "  * to perform the build, execution, and check processes.     *";\
	echo "  *************************************************************";\
	echo
