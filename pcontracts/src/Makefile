#
# Name:      src/Makefile
#
# Purpose:
# This is the top-most PAUL Contracts source subdirectory makefile.
# It is used to manage the build, basic build checking, and cleanup
# of the source files.
#
# Dependencies:
# Refer to the ../setup.sh script for the required environment variables.


include $(PCONTRACTS_HOME)/src/make_test.inc


SUBDIRS = runtime contracts drivers examples tests


.PHONY: all check checkall checksummary clean cleanall

all:
	@echo 1 > $(PC_BUILD_STATUS); \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) all' in $$subdir"; \
	  (cd $$subdir && $(MAKE) all) \
	done;

check:  all
	@echo 1 > $(CHECK_ALL_STATUS); \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) check' in $$subdir"; \
	  (cd $$subdir && $(MAKE) check) \
	done; \
	$(MAKE) checksummary; \
	echo; echo "DONE (check)"; echo

checkall:  all
	@echo 1 > $(CHECK_ALL_STATUS); \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) checkall' in $$subdir"; \
	  (cd $$subdir && $(MAKE) checkall) \
	done; \
	$(MAKE) checksummary; \
	echo; echo "DONE (check)"; echo

checksummary:
	@echo; \
	if [ -f $(PC_BUILD_STATUS) ]; then \
	  RES=`cat $(PC_BUILD_STATUS)`; \
	  if [ "$$RES" -eq "1" ]; then \
	    if [ -f $(CHECK_ALL_STATUS) ]; then \
	      RES=`cat $(CHECK_ALL_STATUS)`; \
	      if [ "$$RES" -eq "1" ]; then \
	        echo "All Suites: PASSED"; \
	      else \
	        echo "All Suites: FAILED"; \
	      fi; \
	    else \
	      echo "All Suites FAILED: $(CHECK_ALL_STATUS) is missing"; \
	    fi; \
	  else \
	    echo "Build: FAILED"; \
	  fi; \
	else \
	  echo "Build FAILED: $(PC_BUILD_STATUS) is missing"; \
	fi; \
	echo


clean: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) clean' in $$subdir"; \
	  (cd $$subdir && $(MAKE) clean) \
	done;

cleanall: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) cleanall' in $$subdir"; \
	  (cd $$subdir && $(MAKE) cleanall) \
	done;
