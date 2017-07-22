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

IF (NOT CMAKE_BUILD_TYPE)
	MESSAGE("SETTING BUILD TYPE TO RELEASE")
	SET(CMAKE_BUILD_TYPE RELEASE)
ENDIF()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -fPIC")


#automated opencv
include(FindPkgConfig)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(JdeRobot REQUIRED)

 include(Deps/glog/CMakeLists.txt)
FIND_PACKAGE(Boost REQUIRED program_options filesystem)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5Gui REQUIRED)
FIND_PACKAGE(Qt5Svg REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)

 SET(QT_INCLUDE_DIRS
	 Qt5Widgets_INCLUDE_DIRS}
	 ${Qt5Core_INCLUDE_DIRS}
	 ${Qt5Gui_INCLUDE_DIRS}
	 ${Qt5Svg_INCLUDE_DIRS}
	 ${Qt5OpenGL_INCLUDE_DIRS}
 )

 set(QT_LIBRARIES
	 ${Qt5Widgets_LIBRARIES}
	 ${Qt5Core_LIBRARIES}
	 ${Qt5Gui_LIBRARIES}
	 ${Qt5Svg_LIBRARIES}
	 ${Qt5OpenGL_LIBRARIES}
)




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




 #darknet
 SET(DARKNET_PATH /home/frivas/devel/mio/darknet)

FIND_PATH( DARKNET_INCLUDE_DIR NAMES DarknetAPI/DarknetAPI.h  PATHS ${DARKNET_PATH}/install/include )
SET(DARKNET_NAMES darknetAPILib darknetExamplesLib darknetLib)
FOREACH(N ${DARKNET_NAMES})
	FIND_LIBRARY( DARKNET_LIBRARIES_${N} NAMES ${N} PATHS ${DARKNET_PATH}/install/lib )
	SET(DARKNET_LIBRARIES ${DARKNET_LIBRARIES} ${DARKNET_LIBRARIES_${N}})

ENDFOREACH()




if (DARKNET_INCLUDE_DIR AND DARKNET_LIBRARIES)
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	SET(DARKNET_ACTIVE ON)
	ADD_DEFINITIONS(-DDARKNET_ACTIVE)
else()
	message("DARKNET_INCLUDE_DIR: ${DARKNET_INCLUDE_DIR}")
	message("DARKNET_LIBRARIES: ${DARKNET_LIBRARIES}")
	MESSAGE(WARNING "DARKNET DISABLE")
endif()


 #glog
 include(FindPackageHandleStandardArgs)

 set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")

 if(WIN32)
	 find_path(GLOG_INCLUDE_DIR glog/logging.h
			 PATHS ${GLOG_ROOT_DIR}/src/windows)
 else()
	 find_path(GLOG_INCLUDE_DIR glog/logging.h
			 PATHS ${GLOG_ROOT_DIR})
 endif()

 if(MSVC)
	 find_library(GLOG_LIBRARY_RELEASE libglog_static
			 PATHS ${GLOG_ROOT_DIR}
			 PATH_SUFFIXES Release)

	 find_library(GLOG_LIBRARY_DEBUG libglog_static
			 PATHS ${GLOG_ROOT_DIR}
			 PATH_SUFFIXES Debug)

	 set(GLOG_LIBRARY optimized ${GLOG_LIBRARY_RELEASE} debug ${GLOG_LIBRARY_DEBUG})
 else()
	 find_library(GLOG_LIBRARY glog
			 PATHS ${GLOG_ROOT_DIR}
			 PATH_SUFFIXES lib lib64)
 endif()

 find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

 if(GLOG_FOUND)
	 set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
	 set(GLOG_LIBRARIES ${GLOG_LIBRARY})
	 message(STATUS "Found glog    (include: ${GLOG_INCLUDE_DIR}, library: ${GLOG_LIBRARY})")
	 mark_as_advanced(GLOG_ROOT_DIR GLOG_LIBRARY_RELEASE GLOG_LIBRARY_DEBUG
			 GLOG_LIBRARY GLOG_INCLUDE_DIR)
 endif()


 add_subdirectory(src)
 add_subdirectory(app)

IF (BUILD_TEST)
	add_subdirectory(test)
ENDIF()