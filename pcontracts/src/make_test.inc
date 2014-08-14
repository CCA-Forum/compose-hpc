#
# Dependencies:
#
# ROOT_DIR   contains the path to the PAUL Contracts root directory
#

RES_DIR=$(ROOT_DIR)/results

BASE_RES_DIR=$(RES_DIR)/base
BASE_ORACLE_DIR=$(SRC_DIR)/tests/data

CHECK_STATUS=.checkStatus
CHECK_ALL_STATUS=$(ROOT_DIR)/$(CHECK_STATUS)

EXAMPLES=-name "*sack*.c*" -o -name "hello*.c*" -o -name "WeightsList*.c*" \
	-o -name "SimpleClass*.c*" -o -name "simple*.c*"

# The following value may need to be customized to your environment
ULIMIT_MAX=unlimited
