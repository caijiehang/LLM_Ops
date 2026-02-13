#include <nanobind/nanobind.h>
#include "device_query.h"
#include "nanobind/nb_defs.h"

namespace nb = nanobind;

NB_MODULE(llm_ops, m){
    nb::class_<cuda_utils::DeviceMemoryInfo>(m,"memoryProps")
        .
}