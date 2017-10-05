cmake_minimum_required(VERSION 3.9)

project(MDPSolver)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES main.cpp)
set(GUROBI_HOME $ENV{GUROBI_HOME})
message("${GUROBI_HOME}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_library( GUROBI_CXX_LIBRARY_1
        NAMES
        libgurobi_g++5.2.a
        PATHS "${GUROBI_HOME}/lib"
        )

message("$ENV{GUROBI_HOME}/lib")

find_library( GUROBI_CXX_LIBRARY_2
        NAMES
        libgurobi75.so
        PATHS "${GUROBI_HOME}/lib"
        )


find_path(GUROBI_INCLUDE_DIR
        NAMES gurobi_c++.h
        PATHS "${GUROBI_HOME}/include"
        )

if(NOT GUROBI_CXX_LIBRARY_1)
    message(FATAL_ERROR "Failed to find Gurobi lib.  Try setting CMAKE_PREFIX_PATH")
endif()

if(NOT GUROBI_CXX_LIBRARY_2)
    message(FATAL_ERROR "Failed to find Gurobi lib 2.  Try setting CMAKE_PREFIX_PATH")
endif()

if(NOT GUROBI_INCLUDE_DIR)
    message(FATAL_ERROR "Failed to find Gurobi headers.  Try setting CMAKE_PREFIX_PATH")
endif()

add_executable(MDPSolver ${SOURCE_FILES})
if(GUROBI_CXX_LIBRARY_1)
    include_directories(${GUROBI_INCLUDE_DIR})
    set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY_1};${GUROBI_CXX_LIBRARY_2}" )
    target_link_libraries(MDPSolver ${GUROBI_LIBRARIES})
endif()
