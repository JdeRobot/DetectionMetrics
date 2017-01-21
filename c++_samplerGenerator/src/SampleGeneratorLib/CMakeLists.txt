SET( SOURCE_FILES RecorderConverter.cpp RecorderConverter.h DepthForegroundSegmentator.cpp DepthForegroundSegmentator.h BoundingValidator.cpp BoundingValidator.h BoundingRect.cpp BoundingRect.h DetectionsValidator.cpp DetectionsValidator.h Logger.cpp Logger.h RectRegion.cpp RectRegion.h ContourRegion.cpp ContourRegion.h Region.h)


include_directories(
    ${JdeRobot_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}
)


add_library (SampleGenerator  ${SOURCE_FILES})
