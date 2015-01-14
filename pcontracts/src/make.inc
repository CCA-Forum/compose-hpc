#
# Name:      src/make.inc
#
# Purpose:
# This make include file defines the core variables for managing the
# PAUL Contracts software.
#
# Dependencies:
# Refer to the ../setup.sh script for the required environment variables.


########## You should NOT have to change anything below this line  ##########
########## except compiler and flags, which may require debugging. ##########

#MAKE=make --no-print-directory
MAKE=make

INCLUDE_DIR=$(PCONTRACTS_HOME)/include
BIN_DIR=$(PCONTRACTS_HOME)/bin
LIB_DIR=$(PCONTRACTS_HOME)/lib
SRC_DIR=$(PCONTRACTS_HOME)/src

CONTRACTS_DIR=$(SRC_DIR)/contracts
CONTRACTS_LIB_BASE=PaulContracts001a
CONTRACTS_LIB=lib$(CONTRACTS_LIB_BASE).a

RT_DIR=$(SRC_DIR)/runtime
RUNTIME_LIB_BASE=ContractsRuntime001a
RUNTIME_LIB=lib$(RUNTIME_LIB_BASE).a

EX_DIR=$(SRC_DIR)/examples

BUILD_STATUS=.buildStatus
PC_BUILD_STATUS=$(PCONTRACTS_HOME)/$(BUILD_STATUS)

#
# Common compiler flags and settings
#
BASE_FLAGS=-g -O3
#BASE_FLAGS=-g -O3 -DDEBUG

BASE_INCLUDES=-I. -I$(INCLUDE_DIR)

CXX=g++
BASE_CXXFLAGS=-Wno-deprecated

