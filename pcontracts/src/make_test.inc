#
# Name:      src/make_test.inc
#
# Purpose:
# This make include file sets variables used to manage testing of the
# basic PAUL Contracts software build.
#
# Dependencies:
# Refer to the ../setup.sh script for the required environment variables.


# The following value may need to be customized to your environment
ULIMIT_MAX=unlimited


########## You should NOT have to change anything below this line ##########

include $(PCONTRACTS_HOME)/src/make.inc

# The following values are dependent on PCONTRACTS_HOME
RES_DIR=$(PCONTRACTS_HOME)/results
BASE_RES_DIR=$(RES_DIR)/base

CHECK_STATUS=.checkStatus
CHECK_ALL_STATUS=$(PCONTRACTS_HOME)/$(CHECK_STATUS)

# The following value is dependent on variable defined in make.inc
BASE_ORACLE_DIR=$(SRC_DIR)/tests/data

# The following value contains all known examples for basic pcontracts
# testing.
EXAMPLES=-name "*sack*.c*" -o -name "hello*.c*" -o -name "WeightsList*.c*" \
	-o -name "SimpleClass*.c*" -o -name "simple*.c*"
