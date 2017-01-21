 cmake_minimum_required(VERSION 2.8)

project(samplerGenerator C CXX)


#check compiller
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	message (version: ${CMAKE_CXX_COMPILER_VERSION})
	if(COMPILER_SUPPORTS_CXX11)
    		set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
	elseif(COMPILER_SUPPORTS_CXX0X)
        #boost have problems with gcc < 4.7 and c++0x
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.7.0")
    		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        endif()
	else()
		    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
	endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -fPIC")


#automated opencv
include(FindPkgConfig)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(JdeRobot REQUIRED)
FIND_PACKAGE(Boost REQUIRED program_options filesystem)



 #manual ICE
FIND_PATH( Ice_INCLUDE_DIR NAMES Ice/Ice.h  PATHS ENV C++LIB ENV)
 
IF( Ice_INCLUDE_DIR )
    FIND_LIBRARY( Ice_LIBRARY1 NAMES Ice PATHS ENV C++LIB ENV PATH PATH_SUFFIXES lib lib64 )
    FIND_LIBRARY( Ice_LIBRARY2 NAMES IceUtil PATHS ENV C++LIB ENV PATH PATH_SUFFIXES lib lib64)
    FIND_LIBRARY( Ice_LIBRARY3 NAMES IceStorm PATHS ENV C++LIB ENV PATH PATH_SUFFIXES lib lib64)
    SET (Ice_LIBRARIES ${Ice_LIBRARY1} ${Ice_LIBRARY2} ${Ice_LIBRARY3})
    IF( Ice_LIBRARIES )
		MESSAGE ("-- Ice found at ${Ice_LIBRARIES}")
		include_directories(${Ice_INCLUDE_DIR})
		link_directories(${Ice_LIBRARIES})
    ENDIF( Ice_LIBRARIES )
ENDIF(Ice_INCLUDE_DIR)

IF(NOT Ice_LIBRARIES)
		MESSAGE ("*** Ice not found")
ENDIF()




add_subdirectory(src)

