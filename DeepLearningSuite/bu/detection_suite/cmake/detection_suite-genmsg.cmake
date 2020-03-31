# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "detection_suite: 2 messages, 0 services")

set(MSG_I_FLAGS "-Idetection_suite:/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(detection_suite_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_custom_target(_detection_suite_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detection_suite" "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" ""
)

get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_custom_target(_detection_suite_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detection_suite" "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" "detection_suite/object"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detection_suite
)
_generate_msg_cpp(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg"
  "${MSG_I_FLAGS}"
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detection_suite
)

### Generating Services

### Generating Module File
_generate_module_cpp(detection_suite
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detection_suite
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(detection_suite_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(detection_suite_generate_messages detection_suite_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_cpp _detection_suite_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_cpp _detection_suite_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detection_suite_gencpp)
add_dependencies(detection_suite_gencpp detection_suite_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detection_suite_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detection_suite
)
_generate_msg_eus(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg"
  "${MSG_I_FLAGS}"
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detection_suite
)

### Generating Services

### Generating Module File
_generate_module_eus(detection_suite
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detection_suite
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(detection_suite_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(detection_suite_generate_messages detection_suite_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_eus _detection_suite_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_eus _detection_suite_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detection_suite_geneus)
add_dependencies(detection_suite_geneus detection_suite_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detection_suite_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detection_suite
)
_generate_msg_lisp(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg"
  "${MSG_I_FLAGS}"
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detection_suite
)

### Generating Services

### Generating Module File
_generate_module_lisp(detection_suite
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detection_suite
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(detection_suite_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(detection_suite_generate_messages detection_suite_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_lisp _detection_suite_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_lisp _detection_suite_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detection_suite_genlisp)
add_dependencies(detection_suite_genlisp detection_suite_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detection_suite_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detection_suite
)
_generate_msg_nodejs(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg"
  "${MSG_I_FLAGS}"
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detection_suite
)

### Generating Services

### Generating Module File
_generate_module_nodejs(detection_suite
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detection_suite
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(detection_suite_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(detection_suite_generate_messages detection_suite_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_nodejs _detection_suite_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_nodejs _detection_suite_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detection_suite_gennodejs)
add_dependencies(detection_suite_gennodejs detection_suite_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detection_suite_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite
)
_generate_msg_py(detection_suite
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg"
  "${MSG_I_FLAGS}"
  "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite
)

### Generating Services

### Generating Module File
_generate_module_py(detection_suite
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(detection_suite_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(detection_suite_generate_messages detection_suite_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/object.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_py _detection_suite_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/hmrishav/DetectionSuite/DeepLearningSuite/detection_suite/msg/objects.msg" NAME_WE)
add_dependencies(detection_suite_generate_messages_py _detection_suite_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detection_suite_genpy)
add_dependencies(detection_suite_genpy detection_suite_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detection_suite_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detection_suite)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detection_suite
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(detection_suite_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detection_suite)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detection_suite
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(detection_suite_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detection_suite)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detection_suite
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(detection_suite_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detection_suite)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detection_suite
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(detection_suite_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detection_suite
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(detection_suite_generate_messages_py std_msgs_generate_messages_py)
endif()
