cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


project(test)

find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package( OpenCV REQUIRED )
find_package(GLOG REQUIRED FlagRegisterer)
include_directories( ${Boost_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${GLOG_INCLUDE_DIRS} )
find_package(GFLAGS REQUIRED) 
include_directories(${GFLAGS_INCLUDE_DIRS})



add_executable(
  output
  test.cpp
  RecorderReader.cpp
  ../readers/DatasetReader.cpp 
  ../../Utils/PathHelper.cpp 
  ../ClassTypeOwn.cpp 
  ../../Common/Sample.cpp 
  ../../Common/EvalMatrix.cpp 
  ../../Regions/RectRegions.cpp 
  ../../Regions/ContourRegions.cpp 
  ../../Regions/RleRegions.cpp
  ../../Regions/maskApi.cpp
  ../../Regions/ContourRegion.cpp
  ../../Regions/RectRegions.cpp
  ../ClassType.cpp
  ../ClassTypeOwn.cpp
)

target_link_libraries(output
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${OpenCV_LIBS} 
  ${GLOG_LIBRARIES}
  ${GLOG_FLAGREGISTERER}
  ${GFLAGS_LIBRARIES}
)

