#include "source/utils/device_query.cuh"

int main()
{
    cuda_utils::print_device_properties(0);
    return 0;
}