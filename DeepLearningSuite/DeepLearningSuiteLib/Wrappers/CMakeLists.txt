set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PATH( DARKNET_INCLUDE_DIR NAMES darknet.h  PATHS ${DARKNET_PATH}/include )

if(UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fPIC; -O3; -ccbin ${CMAKE_CXX_COMPILER})
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -ccbin "\"${CMAKE_CXX_COMPILER}\"")
endif()


#if(USE_GPU)
#cuda_add_library (darknetAPILib ${darknetAPI_SOURCES} ${gpu_SRCS} SHARED)
  #else()
	#add_library (darknetAPILib SHARED ${darknetAPI_SOURCES})
  #endif(USE_GPU)




IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS}
         ${DARKNET_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${CUDA_LIBRARIES}
	 ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY})
        include_directories(${DARKNET_INCLUDE_DIR})
        message("DARKNET_INCLUDE_DIR HERE: ${DARKNET_INCLUDE_DIR}")
        message("CUDA_LIBRARIES HERE: ${CUDA_LIBRARIES}")
        message("CUDA CUBLAS LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
        message("CUDA_curand_LIBRARY: ${CUDA_curand_LIBRARY}")

ENDIF()


SET(Wrappers_SOURCE_FILES
  DarknetAPI
  DarknetAPIConversions.h
  DarknetDetection
  yolo
)


#cuda_add_library (darknetAPILib ${Wrappers_SOURCE_FILES} ${gpu_SRCS} SHARED)
cuda_add_library (WrappersLib ${Wrappers_SOURCE_FILES} ${gpu_SRCS} SHARED)

ADD_LIBRARY(DeepLearningSuite_Wrappers  OBJECT ${Wrappers_SOURCE_FILES})


TARGET_INCLUDE_DIRECTORIES ( DeepLearningSuite_Wrappers PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${config_INCLUDE_DIRS}
  ${depthLib_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${utils_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${DeepLearningSuite_INCLUDE_DIR}
  ${DARKNET_INCLUDE_DIR}
  ${DARKNET_PATH}/src
  ${DARKNET_PATH}/include
  ${DARKNET_PATH}/src
  # ${EXTRA_LIBS}
)

TARGET_LINK_LIBRARIES( DeepLearningSuite_Wrappers 
	darknetLib
	${EXTRA_LIBS})
