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
nanobind_add_module(llm_ops  # 这将生成 cuda_test.cpython-xxx.so
    source/bindings/binding.cpp
    source/host/device_query.cpp
    source/host/launcher.cu
)

# ==========================================
# 4. 设置头文件搜索路径
# ==========================================
target_include_directories(llm_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/source/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/source/kernels
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 将CUDA Runtime动态库链接到程序中
target_link_libraries(llm_ops PRIVATE 
    CUDA::cudart 
)

# 分离编译，开启可重定位设备代码
set_target_properties(llm_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
)

# 2. 生成 .pyi 到 scripts
# 2. 生成 .pyi 到 scripts
nanobind_add_stub(
    llm_ops_stub                 # 目标名称，必须唯一但没有其他实际意义 [cite: 95]
    MODULE llm_ops               # 指定需要被导入并生成存根的模块名称 [cite: 105]
    RECURSIVE                    # 关键！自动遍历模块层次结构并为每个发现的子模块生成独立的存根 
    
    # 在 RECURSIVE 模式下，可以传入多个 OUTPUT 路径。
    # 这主要是为了让 CMake 能够追踪这些生成的文件（例如用于 clean 目标或后续依赖），通常是必填的 [cite: 104, 108, 109, 110]
    OUTPUT 
        "${CMAKE_SOURCE_DIR}/scripts/llm_ops/__init__.pyi"
        "${CMAKE_SOURCE_DIR}/scripts/llm_ops/elementwise.pyi"
        "${CMAKE_SOURCE_DIR}/scripts/llm_ops/gemm.pyi"
        "${CMAKE_SOURCE_DIR}/scripts/llm_ops/m_utils.pyi"
        
    OUTPUT_PATH "${CMAKE_SOURCE_DIR}/scripts"  # 覆盖存根文件写入的基础目录 
    PYTHON_PATH "${CMAKE_SOURCE_DIR}/scripts"  # 告诉 stubgen 在导入时需要搜索的路径 
    DEPENDS llm_ops                            # 声明构建依赖，确保在底层的 .so 编译完成后再执行 [cite: 119, 120]
)