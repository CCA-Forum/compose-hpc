SUBDIRS = examples src tests

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
	done;

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
