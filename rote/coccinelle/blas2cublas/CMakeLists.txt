cmake_minimum_required (VERSION 2.6)
project (Blas2Cublas)

# The version number.
set (B2CB_VERSION_MAJOR 1)
set (B2CB_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/B2CBConfig.h.in"
  "${PROJECT_BINARY_DIR}/include/B2CBConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find TASCELConfig.h
include_directories("${PROJECT_BINARY_DIR}")

SET(ROSE $ENV{ROSE_HOME})
SET(BOOST $ENV{BOOST_HOME})
SET(PAUL $ENV{PAUL_HOME})

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
file(GLOB B2CBOBJ RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/src/*.cpp") # select all .cpp-files

set( B2CBL ${B2CBOBJ} )

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(B2CBLIB STATIC ${B2CBL})

# Set library paths.
link_directories(${ROSE}/lib ${PAUL}/src ${PROJECT_BINARY_DIR}/lib)
# Build the B2CB Executable.
add_executable(B2CB ${PROJECT_SOURCE_DIR}/src/Blas2Cublas.cpp)
# Set libraries to be linked.
target_link_libraries(B2CB B2CBLIB m rose paul fl)

