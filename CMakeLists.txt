cmake_minimum_required(VERSION 3.18)
project(LLM_Ops LANGUAGES CXX CUDA)

# 强制不使用响应文件 (.rsp) 来传递 include 路径
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构（例如 80 是 A100，89 是 4090/Ada）
# 如果不设置，CMake 会尝试检测或报错
set(CMAKE_CUDA_ARCHITECTURES 100) 
find_package(CUDAToolkit REQUIRED)
# 寻找环境中的 python 和 nanobind
# find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
# find_package(nanobind REQUIRED)

# 定义源文件列表，将需要编译的 .cu 文件列在此处
add_executable(cuda_test
    test.cu                        # 你的主入口文件 (必须包含 main 函数)
    source/host/device_query.cpp    # 你依赖的其他 .cu/.cpp 实现文件
)

# # 创建 nanobind 模板
# nanobind_add_module(
#     llm_ops_cuda    # 在 Python 中 import 的模块名
#     NB_STATIC
#     ${SOURCE_FILES}
# )

# ==========================================
# 4. 设置头文件搜索路径
# ==========================================
target_include_directories(cuda_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/utils
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(cuda_test PRIVATE 
    CUDA::cudart 
)

# CUDA 特殊设置
set_target_properties(cuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)