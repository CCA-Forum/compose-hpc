#
# Dependencies: None
#
# ROOT_DIR   contains the path to the PAUL Contracts root directory
#

MAKE=make --no-print-directory

INCLUDE_DIR=$(ROOT_DIR)/include
BIN_DIR=$(ROOT_DIR)/bin
LIB_DIR=$(ROOT_DIR)/lib
SRC_DIR=$(ROOT_DIR)/src

CONTRACTS_DIR=$(SRC_DIR)/contracts
CONTRACTS_LIB_BASE=PaulContracts001a
CONTRACTS_LIB=lib$(CONTRACTS_LIB_BASE).a

RT_DIR=$(SRC_DIR)/runtime
RUNTIME_LIB_BASE=ContractsRuntime001a
RUNTIME_LIB=lib$(RUNTIME_LIB_BASE).a

EX_DIR=$(SRC_DIR)/examples

BUILD_STATUS=.buildStatus
PC_BUILD_STATUS=$(ROOT_DIR)/$(BUILD_STATUS)

#
# Common compiler flags and settings
#
BASE_FLAGS=-g -O3
#BASE_FLAGS=-g -O3 -DDEBUG

BASE_INCLUDES=-I. -I$(INCLUDE_DIR)

CXX=g++
BASE_CXXFLAGS=-Wno-deprecated

