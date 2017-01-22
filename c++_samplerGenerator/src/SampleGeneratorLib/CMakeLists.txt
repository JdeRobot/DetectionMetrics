SET( SOURCE_FILES RecorderConverter.cpp RecorderConverter.h DepthForegroundSegmentator.cpp DepthForegroundSegmentator.h BoundingValidator.cpp BoundingValidator.h BoundingRectGuiMover.cpp BoundingRectGuiMover.h DetectionsValidator.cpp DetectionsValidator.h Logger.cpp Logger.h RectRegions.cpp RectRegions.h ContourRegions.cpp ContourRegions.h Region.h DatasetConverters/DatasetConverter.cpp DatasetConverters/DatasetConverter.h DatasetConverters/DatasetReader.cpp DatasetConverters/DatasetReader.h Sample.cpp Sample.h)


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)


add_library (SampleGenerator  ${SOURCE_FILES})
