# IMPORTANT: Make sure to define your platform with "set(PLATFORM [linux, win32 etc...])" and ensure the enviroment variables JAVA_HOME and CUDA_HOME are set accordingly

cmake_minimum_required(VERSION 3.12.4)
project(deepspeech-jni)

set(CMAKE_CXX_STANDARD 14)

set(CUDA_CONFIGURATION TRUE) # specify whether the jni library should use cuda

set(PLATFORM linux) # define platform here. linux, win32, darwin etc so that $ENV{JAVA_HOME}/include/${PLATFORM} points to the location of "jni_md.h"

if (DEFINED ANDROID_ABI AND CUDA_CONFIGURATION MATCHES TRUE)
    message("Cannot compile cuda jni for android! Building normal cpu configuration instead!")
    set(CUDA_CONFIGURATION FALSE)
endif()

if (DEFINED ANDROID_ABI)
    set(PLATFORM android) # Setting platform to android
endif ()


message("Building for platform: ${PLATFORM}")


if (NOT DEFINED ANDROID_ABI)
    # including JNI
    include_directories($ENV{JAVA_HOME}/include)
    include_directories($ENV{JAVA_HOME}/include/${PLATFORM})
endif()

# include deepspeech header
include_directories(${PROJECT_SOURCE_DIR}/../../) # Including headers of native_client directory

# skip run path. IMPORTANT
set(CMAKE_SKIP_BUILD_RPATH TRUE)

if (CUDA_CONFIGURATION MATCHES TRUE) # if cuda
    add_compile_definitions(CUDA_BUILD_CONFIG)  # defining cuda config compiler constant
    message("Using Cuda build configuration!")
    message("Cuda home directory: \"$ENV{CUDA_HOME}\"")

    # deepspeech-jni with cuda header
    add_library(deepspeech-jni SHARED
            org_mozilla_deepspeech_DeepSpeech.h
            org_mozilla_deepspeech_DeepSpeech.cpp
            org_mozilla_deepspeech_cuda_Cuda.h
            org_mozilla_deepspeech_cuda_Cuda.cpp
            org_mozilla_deepspeech_utils_NativeAccess.h
            org_mozilla_deepspeech_utils_NativeAccess.cpp)

    # including CUDA
    include_directories($ENV{CUDA_HOME}/include)

    # link cuda rt library
    add_library(cudart SHARED IMPORTED)
    set_target_properties(cudart PROPERTIES IMPORTED_LOCATION $ENV{CUDA_HOME}/lib64/libcudart.so) # Make sure this is correct on your machiene
    target_link_libraries(deepspeech-jni cudart)
    set(DS_SUFFIX "cuda")
else ()
    set(DS_SUFFIX "cpu")
    add_compile_definitions(CPU_BUILD_CONFIG) # defining cpu config compiler constant
    message("Using cpu build configuration!")
    # deepspeech-jni without cuda header
    add_library(deepspeech-jni SHARED
            org_mozilla_deepspeech_DeepSpeech.h
            org_mozilla_deepspeech_DeepSpeech.cpp
            org_mozilla_deepspeech_utils_NativeAccess.h
            org_mozilla_deepspeech_utils_NativeAccess.cpp)
endif ()

if (NOT DEFINED ANDROID_ABI)
    set(DS_LIBRARY_LOCATION ${CMAKE_SOURCE_DIR}/libs/${PLATFORM}/libdeepspeech_${DS_SUFFIX}.so)
else ()
    message("Linking deepspeech library for android.")
    set(DS_LIBRARY_LOCATION ${CMAKE_SOURCE_DIR}/libs/${PLATFORM}/${ANDROID_ABI}/libdeepspeech.so)
endif ()

message("Linking deepspeech library located here: \"${DS_LIBRARY_LOCATION}\"")
# deepspeech library
add_library(deepspeech SHARED IMPORTED)
set_target_properties(deepspeech PROPERTIES IMPORTED_LOCATION ${DS_LIBRARY_LOCATION})
target_link_libraries(deepspeech-jni deepspeech)