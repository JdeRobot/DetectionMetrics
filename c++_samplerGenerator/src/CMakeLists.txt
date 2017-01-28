add_subdirectory(SampleGeneratorLib)


SET( SOURCE_FILES main.cpp)

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
    ${JdeRobot_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
)




add_executable (viewer  viewer.cpp)

TARGET_LINK_LIBRARIES(viewer
        SampleGenerator
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        )


add_executable (converter  converter.cpp)

TARGET_LINK_LIBRARIES(converter
        SampleGenerator
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        )
