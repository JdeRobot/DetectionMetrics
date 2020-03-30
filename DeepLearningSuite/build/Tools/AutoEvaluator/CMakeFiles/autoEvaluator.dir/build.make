# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hmrishav/DetectionSuite/DeepLearningSuite

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hmrishav/DetectionSuite/DeepLearningSuite/build

# Include any dependencies generated for this target.
include Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/depend.make

# Include the progress variables for this target.
include Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/progress.make

# Include the compile flags for this target's objects.
include Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/flags.make

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/flags.make
Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o: ../Tools/AutoEvaluator/autoEvaluator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o -c /home/hmrishav/DetectionSuite/DeepLearningSuite/Tools/AutoEvaluator/autoEvaluator.cpp

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.i"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hmrishav/DetectionSuite/DeepLearningSuite/Tools/AutoEvaluator/autoEvaluator.cpp > CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.i

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.s"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hmrishav/DetectionSuite/DeepLearningSuite/Tools/AutoEvaluator/autoEvaluator.cpp -o CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.s

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.requires:

.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.requires

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.provides: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.requires
	$(MAKE) -f Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/build.make Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.provides.build
.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.provides

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.provides.build: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o


# Object files for target autoEvaluator
autoEvaluator_OBJECTS = \
"CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o"

# External object files for target autoEvaluator
autoEvaluator_EXTERNAL_OBJECTS =

Tools/AutoEvaluator/autoEvaluator: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o
Tools/AutoEvaluator/autoEvaluator: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/build.make
Tools/AutoEvaluator/autoEvaluator: DeepLearningSuiteLib/libDeepLearningSuite.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libQt5Svg.so.5.9.5
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.9.5
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_system.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libglog.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
Tools/AutoEvaluator/autoEvaluator: libs/depthLib/libdepthLib.a
Tools/AutoEvaluator/autoEvaluator: devel/lib/libcomm.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroscpp.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole_log4cxx.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole_backend_interface.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_regex.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroscpp_serialization.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libxmlrpcpp.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librostime.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libcpp_common.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_thread.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpthread.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libcv_bridge.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libimage_transport.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libmessage_filters.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libclass_loader.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/libPocoFoundation.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libdl.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroslib.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librospack.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpython2.7.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroscpp.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole_log4cxx.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librosconsole_backend_interface.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_regex.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroscpp_serialization.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libxmlrpcpp.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librostime.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libcpp_common.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_thread.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpthread.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libcv_bridge.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libimage_transport.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libmessage_filters.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libclass_loader.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/libPocoFoundation.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libdl.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/libroslib.so
Tools/AutoEvaluator/autoEvaluator: /opt/ros/melodic/lib/librospack.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libpython2.7.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libboost_system.so
Tools/AutoEvaluator/autoEvaluator: libs/utils/libcolorspacesmm.so
Tools/AutoEvaluator/autoEvaluator: libs/utils/libcolorspaces.a
Tools/AutoEvaluator/autoEvaluator: libs/utils/libcolorspacesshare.so
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_dnn.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_gapi.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_highgui.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_ml.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_objdetect.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_photo.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_stitching.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_video.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_calib3d.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_features2d.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_flann.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_videoio.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_imgproc.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: /usr/local/lib/libopencv_core.so.4.3.0
Tools/AutoEvaluator/autoEvaluator: libs/config/libconfig.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libglog.so
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.5.2
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.9.5
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.9.5
Tools/AutoEvaluator/autoEvaluator: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.9.5
Tools/AutoEvaluator/autoEvaluator: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable autoEvaluator"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/autoEvaluator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/build: Tools/AutoEvaluator/autoEvaluator

.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/build

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/requires: Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/autoEvaluator.cpp.o.requires

.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/requires

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/clean:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator && $(CMAKE_COMMAND) -P CMakeFiles/autoEvaluator.dir/cmake_clean.cmake
.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/clean

Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/depend:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hmrishav/DetectionSuite/DeepLearningSuite /home/hmrishav/DetectionSuite/DeepLearningSuite/Tools/AutoEvaluator /home/hmrishav/DetectionSuite/DeepLearningSuite/build /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator /home/hmrishav/DetectionSuite/DeepLearningSuite/build/Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Tools/AutoEvaluator/CMakeFiles/autoEvaluator.dir/depend

