# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
string(APPEND CMAKE_CXX_FLAGS " /W4")

#onnxruntime-openvino provider
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(OPENCV_ROOTDIR "OpenCV root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
option(OPENCL_LIB "OpenCL lib dir")
option(OPENCL_INCLUDE "OpenCL header dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
endif()
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(OPENCV_ROOTDIR)
  set(OPENCV_FOUND true)
  set(OPENCV_INCLUDE_DIRS "${OPENCV_ROOTDIR}/include")
  set(OPENCV_LIBDIR "${OPENCV_ROOTDIR}/lib")
  file(GLOB OPENCV_DEBUG_LIBRARIES ${OPENCV_LIBDIR}/opencv_imgcodecs*d.lib ${OPENCV_LIBDIR}/opencv_dnn*d.lib ${OPENCV_LIBDIR}/opencv_core*d.lib ${OPENCV_LIBDIR}/opencv_imgproc*d.lib)
  file(GLOB OPENCV_RELEASE_LIBRARIES ${OPENCV_LIBDIR}/opencv_imgcodecs*.lib ${OPENCV_LIBDIR}/opencv_dnn*.lib ${OPENCV_LIBDIR}/opencv_core*.lib ${OPENCV_LIBDIR}/opencv_imgproc*.lib)
  list(FILTER OPENCV_RELEASE_LIBRARIES EXCLUDE REGEX ".*d\\.lib")
endif()

if(OPENCL_LIB AND OPENCL_INCLUDE)
  set(OPENCL_FOUND true)
endif()

if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()

if(OPENCV_FOUND)
  add_subdirectory(squeezenet_classification)
endif()

if(OPENCL_FOUND)
  add_subdirectory(squeezenet_classification_io_buffer)
endif()

add_subdirectory(model-explorer)
