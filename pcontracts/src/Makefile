ROOT_DIR=..
include $(ROOT_DIR)/make.inc

CHECK_ALL_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)

SUBDIRS = runtime examples tests

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
	$(MAKE) checkSummary; \
	echo; echo "DONE (check)"; echo

checkAll:  all
	@echo 1 > $(CHECK_ALL_STATUS); \
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) checkAll' in $$subdir"; \
	  (cd $$subdir && $(MAKE) checkAll) \
	done; \
	$(MAKE) checkSummary; \
	echo; echo "DONE (check)"; echo

checkSummary:
	@echo; \
	if [ -f $(CHECK_ALL_STATUS) ]; then \
	  RES=`cat $(CHECK_ALL_STATUS)`; \
	  if [ "$$RES" -eq "1" ]; then \
	    echo "ALL SUITES: PASSED"; \
	  else \
	    echo "ALL SUITES: FAILED"; \
	  fi; \
	else \
	  echo "ALL SUITES: FAILED: $(CHECK_ALL_STATUS) is missing"; \
	fi; \
	echo

clean: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) clean' in $$subdir"; \
	  (cd $$subdir && $(MAKE) clean) \
	done;

clean-all: cleanAll

cleanAll: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) cleanAll' in $$subdir"; \
	  (cd $$subdir && $(MAKE) cleanAll) \
	done;
