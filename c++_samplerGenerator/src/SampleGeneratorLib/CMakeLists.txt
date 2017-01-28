SET( SOURCE_FILES RecorderConverter.cpp RecorderConverter.h DepthForegroundSegmentator.cpp DepthForegroundSegmentator.h BoundingValidator.cpp BoundingValidator.h BoundingRectGuiMover.cpp BoundingRectGuiMover.h DetectionsValidator.cpp DetectionsValidator.h Logger.cpp Logger.h Regions/RectRegions.cpp Regions/RectRegions.h Regions/ContourRegions.cpp Regions/ContourRegions.h Regions/Regions.h DatasetConverters/DatasetConverter.cpp DatasetConverters/DatasetConverter.h DatasetConverters/DatasetReader.cpp DatasetConverters/DatasetReader.h Sample.cpp Sample.h DatasetConverters/OwnDatasetReader.cpp DatasetConverters/OwnDatasetReader.h DatasetConverters/YoloDatasetConverter.cpp DatasetConverters/YoloDatasetConverter.h DatasetConverters/YoloDatasetReader.cpp DatasetConverters/YoloDatasetReader.h DatasetConverters/ClassType.h DatasetConverters/ClassType.cpp Regions/Region.h Regions/RectRegion.cpp Regions/RectRegion.h Regions/ContourRegion.cpp Regions/ContourRegion.h Detectors/Detector.h)


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)


add_library (SampleGenerator  ${SOURCE_FILES})
