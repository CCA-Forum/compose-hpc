SUBDIRS = runtime examples tests

all:
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) all' in $$subdir"; \
	  (cd $$subdir && $(MAKE) all) \
	done;

check: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) check' in $$subdir"; \
	  (cd $$subdir && $(MAKE) check) \
	done;

clean: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) clean' in $$subdir"; \
	  (cd $$subdir && $(MAKE) clean) \
	done;

clean-all: 
	@list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Calling '$(MAKE) clean-all' in $$subdir"; \
	  (cd $$subdir && $(MAKE) clean-all) \
	done;
