set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)



SET(EXTRA_LIBS ${EXTRA_LIBS} ${DARKNET_LIBRARIES} ${CUDA_LIBRARIES} -L/usr/local/cuda/lib64 cudart cublas curand)



set (SOURCES main.cpp
        mainwindow
        gui/ListViewConfig SamplerGeneratorHandler/Viewer.cpp SamplerGeneratorHandler/Viewer.h gui/TabHandler.cpp gui/TabHandler.h SamplerGeneratorHandler/Converter.cpp SamplerGeneratorHandler/Converter.h SamplerGeneratorHandler/SamplerGenerationHandler.cpp SamplerGeneratorHandler/SamplerGenerationHandler.h gui/Utils.cpp gui/Utils.h SamplerGeneratorHandler/Detector.cpp SamplerGeneratorHandler/Detector.h SamplerGeneratorHandler/Evaluator.cpp SamplerGeneratorHandler/Evaluator.h)


include_directories(
        include_directories(${DARKNET_INCLUDE_DIR})
        ${JdeRobot_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${QT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../src/
        ${CMAKE_CURRENT_LIST_DIR}/../src/SampleGeneratorLib

)



add_executable(samplerGeneratorApp ${SOURCES})


TARGET_LINK_LIBRARIES(samplerGeneratorApp
        SampleGenerator
        ${EXTRA_LIBS}
        ${JdeRobot_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${QT_LIBRARIES}
        )
