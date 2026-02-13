cmake_minimum_required(VERSION 3.18)
project(LLM_Ops LANGUAGES CXX CUDA)

# 强制不使用响应文件 (.rsp) 来传递 include 路径
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
# 自动生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(Python_FIND_VIRTUALENV FIRST)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 架构（例如 80 是 A100，89 是 4090/Ada）
# 如果不设置，CMake 会尝试检测或报错
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)

# 寻找环境中的 python 和 nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# 检测并找到nanobind包
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# 定义源文件列表，将需要编译的 .cu 文件列在此处
nanobind_add_module(LLM_OPS  # 这将生成 cuda_test.cpython-xxx.so
    source/host/device_query.cpp
)

# ==========================================
# 4. 设置头文件搜索路径
# ==========================================
target_include_directories(LLM_OPS PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/utils
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 将CUDA Runtime动态库链接到程序中
target_link_libraries(LLM_OPS PRIVATE 
    CUDA::cudart 
)

# 分离编译，开启可重定位设备代码
set_target_properties(LLM_OPS PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)