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
include libs/utils/CMakeFiles/colorspacesmm.dir/depend.make

# Include the progress variables for this target.
include libs/utils/CMakeFiles/colorspacesmm.dir/progress.make

# Include the compile flags for this target's objects.
include libs/utils/CMakeFiles/colorspacesmm.dir/flags.make

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o: libs/utils/CMakeFiles/colorspacesmm.dir/flags.make
libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o: ../libs/utils/colorspaces/imagecv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o -c /home/hmrishav/DetectionSuite/DeepLearningSuite/libs/utils/colorspaces/imagecv.cpp

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.i"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hmrishav/DetectionSuite/DeepLearningSuite/libs/utils/colorspaces/imagecv.cpp > CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.i

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.s"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hmrishav/DetectionSuite/DeepLearningSuite/libs/utils/colorspaces/imagecv.cpp -o CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.s

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.requires:

.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.requires

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.provides: libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.requires
	$(MAKE) -f libs/utils/CMakeFiles/colorspacesmm.dir/build.make libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.provides.build
.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.provides

libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.provides.build: libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o


# Object files for target colorspacesmm
colorspacesmm_OBJECTS = \
"CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o"

# External object files for target colorspacesmm
colorspacesmm_EXTERNAL_OBJECTS =

libs/utils/libcolorspacesmm.so: libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o
libs/utils/libcolorspacesmm.so: libs/utils/CMakeFiles/colorspacesmm.dir/build.make
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_dnn.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_gapi.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_highgui.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_ml.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_objdetect.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_photo.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_stitching.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_video.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_videoio.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_calib3d.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_features2d.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_flann.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_imgproc.so.4.3.0
libs/utils/libcolorspacesmm.so: /usr/local/lib/libopencv_core.so.4.3.0
libs/utils/libcolorspacesmm.so: libs/utils/CMakeFiles/colorspacesmm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hmrishav/DetectionSuite/DeepLearningSuite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcolorspacesmm.so"
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colorspacesmm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libs/utils/CMakeFiles/colorspacesmm.dir/build: libs/utils/libcolorspacesmm.so

.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/build

libs/utils/CMakeFiles/colorspacesmm.dir/requires: libs/utils/CMakeFiles/colorspacesmm.dir/colorspaces/imagecv.cpp.o.requires

.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/requires

libs/utils/CMakeFiles/colorspacesmm.dir/clean:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils && $(CMAKE_COMMAND) -P CMakeFiles/colorspacesmm.dir/cmake_clean.cmake
.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/clean

libs/utils/CMakeFiles/colorspacesmm.dir/depend:
	cd /home/hmrishav/DetectionSuite/DeepLearningSuite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hmrishav/DetectionSuite/DeepLearningSuite /home/hmrishav/DetectionSuite/DeepLearningSuite/libs/utils /home/hmrishav/DetectionSuite/DeepLearningSuite/build /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils /home/hmrishav/DetectionSuite/DeepLearningSuite/build/libs/utils/CMakeFiles/colorspacesmm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libs/utils/CMakeFiles/colorspacesmm.dir/depend

