cmake_minimum_required(VERSION 3.10)

set(JAVA_AWT_INCLUDE_PATH NotNeeded)

set(JAVA_HOME /usr/lib/jvm/java-1.11.0-openjdk-amd64)

find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

include_directories( ${CMAKE_JAVA_} /usr/lib/jvm/java-1.11.0-openjdk-amd64/include /usr/lib/jvm/java-1.11.0-openjdk-amd64/include/linux )

add_library( deepspeech-jni SHARED ../jni/deepspeech_wrap.cpp )

find_library(deepspeech-lib NAMES deepspeech PATHS ${CMAKE_SOURCE_DIR}/libs/x86_64/ REQUIRED)
message(STATUS ${deepspeech-lib})

add_custom_command( TARGET deepspeech-jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/libs/x86_64/libdeepspeech.so ${CMAKE_SOURCE_DIR}/libdeepspeech.so )

target_link_libraries( deepspeech-jni ${deepspeech-lib} )