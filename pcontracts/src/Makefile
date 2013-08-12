ROOT_DIR=..
include $(ROOT_DIR)/src/make_test.inc

SUBDIRS = runtime contracts drivers examples tests


all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
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
	if [ -f $(CHECK_ALL_STATUS) ]; then \
	  RES=`cat $(CHECK_ALL_STATUS)`; \
	  if [ "$$RES" -eq "1" ]; then \
	    echo "All Suites: PASSED"; \
	  else \
	    echo "All Suites: FAILED"; \
	  fi; \
	else \
	  echo "All SUITESSuites FAILED: $(CHECK_ALL_STATUS) is missing"; \
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
