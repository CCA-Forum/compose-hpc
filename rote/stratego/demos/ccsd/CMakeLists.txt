cmake_minimum_required (VERSION 2.6)
project (CCSD)

# The version number.
set (CCSD_VERSION_MAJOR 1)
set (CCSD_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

INCLUDE(CTest)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/CCSDConfig.h.in"
  "${PROJECT_BINARY_DIR}/include/CCSDConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find CCSDConfig.h
include_directories("${PROJECT_BINARY_DIR}")

SET(ROSE $ENV{ROSE_HOME})
SET(BOOST $ENV{BOOST_HOME})
SET(PAUL $ENV{PAUL_HOME})

#message(${PAUL})
#message(${ROSE})

# Specify Header Include Paths 
set(Includes
    ${PROJECT_SOURCE_DIR}/src/
    ${ROSE}/include
    ${BOOST}/include
    ${PAUL}/src/
    )

# Set Header Include paths.
include_directories(${Includes})

# Compile the code.
file(GLOB CCSDOBJ RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/src/*.cpp") # select all .cpp-files

set( CCSDL ${CCSDOBJ} )

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(CCSDLIB STATIC ${CCSDL})

# Set library paths.
link_directories(${ROSE}/lib ${PAUL}/src ${PROJECT_BINARY_DIR}/lib)
# Build the CCSD Executable.
add_executable(CCSD ${PROJECT_SOURCE_DIR}/src/ccsd.cpp)
# Set libraries to be linked.
target_link_libraries(CCSD CCSDLIB m rose paul fl)

# Create directory to store test results
SET(TestDir ${PROJECT_BINARY_DIR}/Testing/testRuns)
execute_process(COMMAND mkdir -p "${TestDir}")
execute_process(COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/src/ccsd.paulconf ${TestDir})

# Testing

# Select all tests
file(GLOB Testsuite RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tests/*.f90")

# Run each test from the testsuite
foreach (test ${Testsuite})
    get_filename_component(Testname ${test} NAME_WE)
    add_test(${Testname} python ${PROJECT_SOURCE_DIR}/Test.py "${Testname}"
                 "${PROJECT_BINARY_DIR}/CCSD" "${PROJECT_SOURCE_DIR}/${test}" "${TestDir}")
endforeach ()

