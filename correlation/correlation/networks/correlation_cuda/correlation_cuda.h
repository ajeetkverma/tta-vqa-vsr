#include <torch/script.h>

int64_t correlation_forward_cuda(at::Tensor input1,
                                 at::Tensor input2,
                                 at::Tensor rInput1,
                                 at::Tensor rInput2,
                                 at::Tensor output,
                                 int64_t pad_size,
                                 int64_t kernel_size,
                                 int64_t max_displacement,
                                 int64_t stride1,
                                 int64_t stride2,
                                 int64_t corr_type_multiply);
