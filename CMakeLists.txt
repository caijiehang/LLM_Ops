cmake_minimum_required(VERSION 3.18)
project(LLM_Ops LANGUAGES CXX CUDA)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构（例如 80 是 A100，89 是 4090/Ada）
# 如果不设置，CMake 会尝试检测或报错
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 100) 
endif()

# 寻找环境中的 python 和 nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind REQUIRED)

# 定义源文件列表，将需要编译的 .cu 文件列在此处
set(SOURCE_FILES
    source/host/device_query.cu
)

# 创建 nanobind 模板
nanobind_add_module(
    llm_ops_cuda    # 在 Python 中 import 的模块名
    NB_STATIC
    ${SOURCE_FILES}
)

# 设置包含路径
target_include_directories(llm_ops_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/source/utils

)

# CUDA 特殊设置
set_target_properties(llm_ops_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # 如果你的算子跨文件调用，建议开启
    POSITION_INDEPENDENT_CODE ON    # 必须开启以生成 .so/.pyd 文件
)