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
include detection_suite/CMakeFiles/code.dir/depend.make

# Include the progress variables for this target.
include detection_suite/CMakeFiles/code.dir/progress.make

# Include the compile flags for this target's objects.
include detection_suite/CMakeFiles/code.dir/flags.make

detection_suite/CMakeFiles/code.dir/src/code.cpp.o: detection_suite/CMakeFiles/code.dir/flags.make
detection_suite/CMakeFiles/code.dir/src/code.cpp.o: ../detection_suite/src/code.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object detection_suite/CMakeFiles/code.dir/src/code.cpp.o"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/code.dir/src/code.cpp.o -c /home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/src/code.cpp

detection_suite/CMakeFiles/code.dir/src/code.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code.dir/src/code.cpp.i"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/src/code.cpp > CMakeFiles/code.dir/src/code.cpp.i

detection_suite/CMakeFiles/code.dir/src/code.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code.dir/src/code.cpp.s"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/src/code.cpp -o CMakeFiles/code.dir/src/code.cpp.s

detection_suite/CMakeFiles/code.dir/src/code.cpp.o.requires:

.PHONY : detection_suite/CMakeFiles/code.dir/src/code.cpp.o.requires

detection_suite/CMakeFiles/code.dir/src/code.cpp.o.provides: detection_suite/CMakeFiles/code.dir/src/code.cpp.o.requires
	$(MAKE) -f detection_suite/CMakeFiles/code.dir/build.make detection_suite/CMakeFiles/code.dir/src/code.cpp.o.provides.build
.PHONY : detection_suite/CMakeFiles/code.dir/src/code.cpp.o.provides

detection_suite/CMakeFiles/code.dir/src/code.cpp.o.provides.build: detection_suite/CMakeFiles/code.dir/src/code.cpp.o


detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o: detection_suite/CMakeFiles/code.dir/flags.make
detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o: detection_suite/code_autogen/mocs_compilation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o -c /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite/code_autogen/mocs_compilation.cpp

detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.i"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite/code_autogen/mocs_compilation.cpp > CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.i

detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.s"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite/code_autogen/mocs_compilation.cpp -o CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.s

detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.requires:

.PHONY : detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.requires

detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.provides: detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.requires
	$(MAKE) -f detection_suite/CMakeFiles/code.dir/build.make detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.provides.build
.PHONY : detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.provides

detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.provides.build: detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o


# Object files for target code
code_OBJECTS = \
"CMakeFiles/code.dir/src/code.cpp.o" \
"CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o"

# External object files for target code
code_EXTERNAL_OBJECTS =

devel/lib/detection_suite/code: detection_suite/CMakeFiles/code.dir/src/code.cpp.o
devel/lib/detection_suite/code: detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o
devel/lib/detection_suite/code: detection_suite/CMakeFiles/code.dir/build.make
devel/lib/detection_suite/code: DeepLearningSuiteLib/libDeepLearningSuite.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libQt5Svg.so.5.9.5
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.9.5
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libglog.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpython3.6m.so
devel/lib/detection_suite/code: libs/depthLib/libdepthLib.a
devel/lib/detection_suite/code: devel/lib/libcomm.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroscpp.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole_log4cxx.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole_backend_interface.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroscpp_serialization.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libxmlrpcpp.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librostime.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libcpp_common.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libcv_bridge.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libimage_transport.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libmessage_filters.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libclass_loader.so
devel/lib/detection_suite/code: /usr/lib/libPocoFoundation.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libdl.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroslib.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librospack.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpython2.7.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroscpp.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole_log4cxx.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librosconsole_backend_interface.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroscpp_serialization.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libxmlrpcpp.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librostime.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libcpp_common.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libcv_bridge.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libimage_transport.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libmessage_filters.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libclass_loader.so
devel/lib/detection_suite/code: /usr/lib/libPocoFoundation.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libdl.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/libroslib.so
devel/lib/detection_suite/code: /opt/ros/melodic/lib/librospack.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libpython2.7.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/detection_suite/code: libs/utils/libcolorspacesmm.so
devel/lib/detection_suite/code: libs/utils/libcolorspaces.a
devel/lib/detection_suite/code: libs/utils/libcolorspacesshare.so
devel/lib/detection_suite/code: /usr/local/lib/libopencv_dnn.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_gapi.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_highgui.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_ml.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_objdetect.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_photo.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_stitching.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_video.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_calib3d.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_features2d.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_flann.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_videoio.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_imgproc.so.4.3.0
devel/lib/detection_suite/code: /usr/local/lib/libopencv_core.so.4.3.0
devel/lib/detection_suite/code: libs/config/libconfig.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libglog.so
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.5.2
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.9.5
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.9.5
devel/lib/detection_suite/code: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.9.5
devel/lib/detection_suite/code: detection_suite/CMakeFiles/code.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../devel/lib/detection_suite/code"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/code.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
detection_suite/CMakeFiles/code.dir/build: devel/lib/detection_suite/code

.PHONY : detection_suite/CMakeFiles/code.dir/build

detection_suite/CMakeFiles/code.dir/requires: detection_suite/CMakeFiles/code.dir/src/code.cpp.o.requires
detection_suite/CMakeFiles/code.dir/requires: detection_suite/CMakeFiles/code.dir/code_autogen/mocs_compilation.cpp.o.requires

.PHONY : detection_suite/CMakeFiles/code.dir/requires

detection_suite/CMakeFiles/code.dir/clean:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite && $(CMAKE_COMMAND) -P CMakeFiles/code.dir/cmake_clean.cmake
.PHONY : detection_suite/CMakeFiles/code.dir/clean

detection_suite/CMakeFiles/code.dir/depend:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hmrishav/DetectionSuite/DeepLearningSuite /home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite /home/hmrishav/DetectionSuite/DeepLearningSuite/build /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite /home/hmrishav/DetectionSuite/DeepLearningSuite/build/detection_suite/CMakeFiles/code.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : detection_suite/CMakeFiles/code.dir/depend

