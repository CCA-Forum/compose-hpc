SUBDIRS = src

all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) all) \
	done;

docs: clean-docs
	@ln -s ${ROSE_HOME} rose; \
	doxygen

check: all
	@echo 1 > .checkStatus; \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) check) \
	done; \
	echo; \
	if [ -f .checkStatus ]; then \
	  RES=`cat .checkStatus`; \
	  if [ $$RES -eq 1 ]; then \
	    echo "CHECKS PROBABLY PASSED"; \
	  else \
	    echo "CHECKS FAILED"; \
	  fi; \
	else \
	  echo "CHECKS FAILED (Missing .checkStatus file)"; \
	fi; \
	echo; \
	echo "  *************************************************************";\
	echo "  * Standard out messages and generated source files must be  *";\
	echo "  * reviewed manually since only some executions end in 'pass *";\
	echo "  * /fail' comparisons.                                       *";\
	echo "  *                                                           *";\
	echo "  * In addition, only one level of indirection is supported.  *";\
	echo "  * That is, 'make check' of instrumented source files, found *";\
	echo "  * in the results subdirectory, must be performed manually.  *";\
	echo "  * Exercising instrumented checks requires the software be   *";\
	echo "  * built with 'PCEFLAGS=-DPAUL_CONTRACTS'.  It can be added  *";\
	echo "  * during check (ie, 'make PCEFLAGS=-DPAUL_CONTRACTS check') *";\
	echo "  * to perform the build, execution, and check processes.     *";\
	echo "  *************************************************************";\
	echo; rm -f .checkStatus

clean: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) clean) \
	done;

clean-all:  clean-docs
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) clean-all) \
	done;

clean-docs:
	@rm -rf rose docs
