/* Copyright 2020 K|Lens GmbH */

#include "correlation.h"
#include <iostream>
#ifdef PYTHON_BUILD
#include <torch/extension.h>
#endif

using namespace torch;
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define WITHIN_BOUNDS3(val1, val2, bound1, bound2) (val1 >= 0 && val1 < bound1 && val2 >= 0 && val2 < bound2)
#define WITHIN_BOUNDS2(x, bound) (x >=0 && x < bound)


template <typename scalar_t>
static void correlate_patch(
    TensorAccessor<scalar_t, 3> f1_acc,
    TensorAccessor<scalar_t, 3> f2_acc,
    TensorAccessor<scalar_t, 3> out_acc,
    int64_t h, /* height cordinate number */
    int64_t w, /* width cordinate number */
    int64_t max_displacement,
    int64_t stride2
){
    /*
    We are in the h,w position of both feature maps.
    */
    int64_t f1c = f1_acc.size(0);
    int64_t f1h = f1_acc.size(1);
    int64_t f1w = f1_acc.size(2);

    /* Indicies that define the extents of the window. */
    int64_t win_starth, win_endh, win_startw, win_endw;
    win_starth = h - max_displacement;
    win_endh = h + max_displacement + 1;
    win_startw = w - max_displacement;
    win_endw = w + max_displacement + 1;

    int64_t c, ph, pw, outpc = 0;

    for ( ph = win_starth; ph < win_endh; ph+=stride2){
        for ( pw = win_startw; pw < win_endw; pw+=stride2){
            if ( WITHIN_BOUNDS3(ph, pw, f1h, f1w /* better to have f2 here maybe */)){
                // We are in the window now.
                scalar_t outval = 0.0;
                for (c = 0; c < f1c; c++){
                    outval += f1_acc[c][h][w] * f2_acc[c][ph][pw];
                }
                // TODO: Optimization: We can get this from ph and pw. This should be ph * (win_endh - win_starth)
                // Output channel index.
                // outpc = (( ph - win_starth)/stride2) * max_displacement + (pw - win_startw)/stride2;
                // std::cout<<"outpc = "<<outpc<<std::endl;
                scalar_t *access_ptr = &out_acc[outpc][h][w];
                *access_ptr = outval / scalar_t(c);
            }
            outpc++;
        }
    }
}

torch::Tensor correlation_cpp_forward(
    torch::Tensor f1,
    torch::Tensor f2,
    int64_t pad_size,
    int64_t kernel_size,
    int64_t max_displacement,
    int64_t stride1,
    int64_t stride2){

        if (kernel_size != 1){
            std::cout<< "kernel_size = "<<kernel_size<<" not supported\n"
                        "Please use kernel_size of 1";;
        }
        if (pad_size != 20){
            std::cout<< "Pad size = "<<pad_size<<" not supported\n"
                        "Please use pad_size of 20";
        }
        const auto f1b = f1.size(0);
        const auto f1h = f1.size(2);
        const auto f1w = f1.size(3);

        // Works as long as stride2 and max_displacement agree.
        const auto outc = ( 2*(max_displacement / stride2) + 1) * ( 2*(max_displacement / stride2) + 1);
        const auto outb = f1b;
        const auto outh = f1h;
        const auto outw = f1w;

        auto output = at::zeros({outb, outc, outh, outw}, f1.options());

        int64_t n, h, w;
        #ifdef _OPENMP
        #pragma omp parallel for private(n, h, w) collapse(2)
        #endif
        for (n = 0; n < outb; ++n) {
            for(h = 0; h < outh; h+=stride1){
                for(w = 0; w < outw; w+=stride1){
                    AT_DISPATCH_FLOATING_TYPES(f1.scalar_type(), "correlation_forward_cpp", ([&]{
                        auto f1_acc = f1.accessor<scalar_t, 4>();
                        auto f2_acc = f2.accessor<scalar_t, 4>();
                        auto out_acc = output.accessor<scalar_t, 4>();

                        correlate_patch(
                                f1_acc[n],
                                f2_acc[n],
                                out_acc[n],
                                h,
                                w,
                                max_displacement,
                                stride2);
                    }));
                }
            }
        }
        return output;
    }


#ifdef PYTHON_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_cpp_forward, "Spatial Correlation Sampler Forward");
//   m.def("backward", &correlation_cpp_backward, "Spatial Correlation Sampler backward");
}
static auto registry = torch::RegisterOperators("my_ops::correlation_cpp_forward", correlation_cpp_forward);
#endif
