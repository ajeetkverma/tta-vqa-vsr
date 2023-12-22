#include<torch/script.h>

torch::Tensor correlation_cpp_forward(
    torch::Tensor f1,
    torch::Tensor f2,
    int64_t pad_size,
    int64_t kernel_size,
    int64_t max_displacement,
    int64_t stride1,
    int64_t stride2);
