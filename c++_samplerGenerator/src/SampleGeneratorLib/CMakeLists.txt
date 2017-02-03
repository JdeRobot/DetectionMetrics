SET( SOURCE_FILES RecorderConverter
       DepthForegroundSegmentator
        BoundingValidator
        BoundingRectGuiMover
        DetectionsValidator
        Logger
        Regions/RectRegions
        Regions/ContourRegions
        Regions/Regions.h
        DatasetConverters/DatasetConverter
        DatasetConverters/DatasetReader
        Sample
        DatasetConverters/OwnDatasetReader
        DatasetConverters/YoloDatasetWriter
        DatasetConverters/YoloDatasetReader
        DatasetConverters/ClassTypeVoc
        Regions/Region
        Regions/RectRegion
        Regions/ContourRegion
        Detectors/Detector.h
        DatasetConverters/SpinelloDatasetReader
        FrameworkEvaluator/FrameworkInferencer
        Utils/StringHandler.h
        Utils/Normalizations
        FrameworkEvaluator/Statistics
        DatasetConverters/ClassTypeCoco
        FrameworkEvaluator/DetectionsEvaluator
        FrameworkEvaluator/MassInferencer
        DatasetConverters/ClassTypeOwn
        DatasetConverters/ClassType
        )


IF(DARKNET_ACTIVE)
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/DarknetInferencer )
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)


add_library (SampleGenerator  ${SOURCE_FILES})
