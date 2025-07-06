cmake_minimum_required(VERSION 3.10)
project(2023e NONE)  # NONE表示不涉及编程语言编译

# 查找Python解释器
find_package(PythonInterp 3 REQUIRED)

# 查找所需包(仅检查是否存在)
find_package(OpenCV REQUIRED)
find_library(WIRINGPI_LIBRARIES NAMES wiringPi wiringOP)

# 指定src目录下的Python文件
set(MAIN_PYTHON_FILE ${CMAKE_SOURCE_DIR}/src/opencv.py)
if(NOT EXISTS ${MAIN_PYTHON_FILE})
    message(FATAL_ERROR "Python not found: ${MAIN_PYTHON_FILE}")
endif()

# 创建运行目标
add_custom_target(run
    COMMAND ${PYTHON_EXECUTABLE} ${MAIN_PYTHON_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src
)

# 输出配置信息
message(STATUS "Python : ${PYTHON_EXECUTABLE}")
message(STATUS "OpenCV : ${OpenCV_VERSION}")
message(STATUS "Python : ${MAIN_PYTHON_FILE}")
