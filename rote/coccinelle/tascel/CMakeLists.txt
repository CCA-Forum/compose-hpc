cmake_minimum_required (VERSION 2.6)
project (TASCEL)

# The version number.
set (TASCEL_VERSION_MAJOR 1)
set (TASCEL_VERSION_MINOR 0)

set (CMAKE_BUILD_TYPE Debug)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TascelConfig.h.in"
  "${PROJECT_BINARY_DIR}/include/TascelConfig.h"
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
file(GLOB TASCELOBJ RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/src/*.cpp") # select all .cpp-files

set( TASCELL ${TASCELOBJ} )

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(TASCELLIB STATIC ${TASCELL})

# Set library paths.
link_directories(${ROSE}/lib ${PAUL}/src ${PROJECT_BINARY_DIR}/lib)
# Build the TASCEL Executable.
add_executable(TASCEL ${PROJECT_SOURCE_DIR}/src/Tascel.cpp)
# Set libraries to be linked.
target_link_libraries(TASCEL TASCELLIB m rose paul fl)

