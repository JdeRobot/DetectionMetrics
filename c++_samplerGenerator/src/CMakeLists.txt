add_subdirectory(SampleGeneratorLib)


SET(EXTRA_LIBS "")
IF (DARKNET_ACTIVE)
    SET(EXTRA_LIBS ${EXTRA_LIBS} ${DARKNET_LIBRARIES} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 cudart cublas curand)
    include_directories(${DARKNET_INCLUDE_DIR})
endif()



SET( SOURCE_FILES generator.cpp)


add_executable (sampler  ${SOURCE_FILES})

include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/SampleGeneratorLib
)

TARGET_LINK_LIBRARIES(sampler
    SampleGenerator
        ${EXTRA_LIBS}
    ${JdeRobot_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )




add_executable (viewer  viewer.cpp)

TARGET_LINK_LIBRARIES(viewer
        SampleGenerator
        ${EXTRA_LIBS}

        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )


add_executable (converter  converter.cpp)

TARGET_LINK_LIBRARIES(converter
        SampleGenerator
        ${EXTRA_LIBS}

        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )


add_executable (detector detector.cpp)

TARGET_LINK_LIBRARIES(detector
        SampleGenerator
        ${EXTRA_LIBS}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )


add_executable (evaluator evaluator.cpp)

TARGET_LINK_LIBRARIES(evaluator
        SampleGenerator
        ${EXTRA_LIBS}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )

add_executable (splitter splitter.cpp)

TARGET_LINK_LIBRARIES(splitter
        SampleGenerator
        ${EXTRA_LIBS}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ssl
        crypto
        )
