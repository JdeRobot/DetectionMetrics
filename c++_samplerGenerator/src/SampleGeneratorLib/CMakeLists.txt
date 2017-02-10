SET( SOURCE_FILES
        GenerationUtils/RecorderConverter
        GenerationUtils/DepthForegroundSegmentator
        GenerationUtils/BoundingValidator
        GenerationUtils/BoundingRectGuiMover
        GenerationUtils/DetectionsValidator
        Utils/Logger
        Regions/RectRegions
        Regions/ContourRegions
        Regions/Regions.h
        DatasetConverters/DatasetConverter
        DatasetConverters/DatasetReader
        Sample
        DatasetConverters/OwnDatasetReader
        DatasetConverters/YoloDatasetWriter
        DatasetConverters/YoloDatasetReader
        DatasetConverters/ClassTypeGeneric
        Regions/Region
        Regions/RectRegion
        Regions/ContourRegion
        Detectors/Detector.h
        DatasetConverters/SpinelloDatasetReader
        FrameworkEvaluator/FrameworkInferencer
        Utils/StringHandler.h
        Utils/Normalizations
        FrameworkEvaluator/ClassStatistics
        FrameworkEvaluator/DetectionsEvaluator
        FrameworkEvaluator/MassInferencer
        DatasetConverters/ClassTypeOwn
        DatasetConverters/ClassType
        Utils/Configuration
        Utils/Key
        Utils/SampleGenerationApp
        DatasetConverters/GenericDatasetReader
        FrameworkEvaluator/GenericInferencer
        DatasetConverters/DatasetWriter
        DatasetConverters/OwnDatasetWriter
        DatasetConverters/GenericDatasetWriter
        Utils/StatsUtils
        )


IF(DARKNET_ACTIVE)
    SET( SOURCE_FILES ${SOURCE_FILES} FrameworkEvaluator/DarknetInferencer)
    include_directories(${DARKNET_INCLUDE_DIR})
ENDIF()


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)


add_library (SampleGenerator  ${SOURCE_FILES})
