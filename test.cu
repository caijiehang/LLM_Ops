#include "source/utils/device_query.h"

int main()
{
    cuda_utils::print_device_properties(0);
    return 0;
}