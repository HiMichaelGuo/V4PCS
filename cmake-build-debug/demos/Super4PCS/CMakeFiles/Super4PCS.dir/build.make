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
CMAKE_COMMAND = /work/juan/mozilla_duj0/clion-2018.1.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /work/juan/mozilla_duj0/clion-2018.1.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /work/juan/proj_others/V4PCS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /work/juan/proj_others/V4PCS/cmake-build-debug

# Include any dependencies generated for this target.
include demos/Super4PCS/CMakeFiles/Super4PCS.dir/depend.make

# Include the progress variables for this target.
include demos/Super4PCS/CMakeFiles/Super4PCS.dir/progress.make

# Include the compile flags for this target's objects.
include demos/Super4PCS/CMakeFiles/Super4PCS.dir/flags.make

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o: demos/Super4PCS/CMakeFiles/Super4PCS.dir/flags.make
demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o: ../demos/Super4PCS/super4pcs_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/work/juan/proj_others/V4PCS/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o"
	cd /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o -c /work/juan/proj_others/V4PCS/demos/Super4PCS/super4pcs_test.cc

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Super4PCS.dir/super4pcs_test.cc.i"
	cd /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /work/juan/proj_others/V4PCS/demos/Super4PCS/super4pcs_test.cc > CMakeFiles/Super4PCS.dir/super4pcs_test.cc.i

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Super4PCS.dir/super4pcs_test.cc.s"
	cd /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /work/juan/proj_others/V4PCS/demos/Super4PCS/super4pcs_test.cc -o CMakeFiles/Super4PCS.dir/super4pcs_test.cc.s

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.requires:

.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.requires

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.provides: demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.requires
	$(MAKE) -f demos/Super4PCS/CMakeFiles/Super4PCS.dir/build.make demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.provides.build
.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.provides

demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.provides.build: demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o


# Object files for target Super4PCS
Super4PCS_OBJECTS = \
"CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o"

# External object files for target Super4PCS
Super4PCS_EXTERNAL_OBJECTS =

demos/Super4PCS/Super4PCS: demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o
demos/Super4PCS/Super4PCS: demos/Super4PCS/CMakeFiles/Super4PCS.dir/build.make
demos/Super4PCS/Super4PCS: src/super4pcs/algorithms/libsuper4pcs_algo.a
demos/Super4PCS/Super4PCS: src/super4pcs/io/libsuper4pcs_io.a
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
demos/Super4PCS/Super4PCS: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
demos/Super4PCS/Super4PCS: demos/Super4PCS/CMakeFiles/Super4PCS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/work/juan/proj_others/V4PCS/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Super4PCS"
	cd /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Super4PCS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/Super4PCS/CMakeFiles/Super4PCS.dir/build: demos/Super4PCS/Super4PCS

.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/build

demos/Super4PCS/CMakeFiles/Super4PCS.dir/requires: demos/Super4PCS/CMakeFiles/Super4PCS.dir/super4pcs_test.cc.o.requires

.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/requires

demos/Super4PCS/CMakeFiles/Super4PCS.dir/clean:
	cd /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS && $(CMAKE_COMMAND) -P CMakeFiles/Super4PCS.dir/cmake_clean.cmake
.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/clean

demos/Super4PCS/CMakeFiles/Super4PCS.dir/depend:
	cd /work/juan/proj_others/V4PCS/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /work/juan/proj_others/V4PCS /work/juan/proj_others/V4PCS/demos/Super4PCS /work/juan/proj_others/V4PCS/cmake-build-debug /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS /work/juan/proj_others/V4PCS/cmake-build-debug/demos/Super4PCS/CMakeFiles/Super4PCS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/Super4PCS/CMakeFiles/Super4PCS.dir/depend

