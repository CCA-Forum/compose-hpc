SUBDIRS = src

all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) all) \
	done;

docs: clean-docs
	@ln -s ${ROSE_HOME} rose; \
	doxygen

check: all
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  (cd $$subdir && $(MAKE) check) \
	done; \
	echo; \
	echo "  *************************************************************";\
	echo "  * Standard out messages and generated source files must be  *";\
	echo "  * reviewed manually since only some executions end in 'pass *";\
	echo "  * /fail' comparisons and results are NOT rolled up into an  *";\
	echo "  * overall PASS/FAIL indication.                             *";\
	echo "  *                                                           *";\
	echo "  * In addition, only one level of indirection is supported.  *";\
	echo "  * That is, 'make check' of instrumented source files, found *";\
	echo "  * in the results subdirectory, must be performed manuually. *";\
	echo "  *************************************************************";\
	echo

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
