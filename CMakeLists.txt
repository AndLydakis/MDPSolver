set(CMAKE_CXX_COMPILER g++-4.9)
cmake_minimum_required(VERSION 3.9)

project(MDPSolver)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_library( GUROBI_CXX_LIBRARY_1
        NAMES
        libgurobi_c++.a
        PATHS "${GUROBI_HOME}/lib"
        )
find_library( GUROBI_CXX_LIBRARY_2
        NAMES
        libgurobi75.so
        PATHS "${GUROBI_HOME}/lib"
        )

find_path(GUROBI_INCLUDE_DIR
        NAMES gurobi_c++.h
        PATHS "${GUROBI_HOME}/include"
        )

if(NOT GUROBI_CXX_LIBRARY)
    message(FATAL_ERROR "Failed to find Gurobi lib.  Try setting CMAKE_PREFIX_PATH")
endif()
add_executable(MDPSolver ${SOURCE_FILES})
if(GUROBI_CXX_LIBRARY)
    include_directories(${GUROBI_INCLUDE_DIR})
    set(GUROBI_LIBRARIES "${GUROBI_LIBRARY};${GUROBI_CXX_LIBRARY}" )
    target_link_libraries(MDPSolver ${GUROBI_CXX_LIBRARY_1})
    target_link_libraries(MDPSolver ${GUROBI_CXX_LIBRARY_2})
#    target_link_libraries(MDPSolver /opt/gurobi751/linux64/libgurobi_c++.a)
    message(STATUS "Gurobi found ${GUROBI_CXX_LIBRARY}")
endif()
