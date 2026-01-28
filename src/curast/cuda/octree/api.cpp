#include <torch/extension.h>
#include "api.h"
#include "config.h"
#include "forward.h"

static std::function<char* (size_t N)> resize_functional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long) N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OctreeVoxelRasterizer::rasterize_cuda(
    const torch::Tensor& background,
    const torch::Tensor& positions,
    const torch::Tensor& colors,
    const torch::Tensor& densities,
    const torch::Tensor& depths,
    const float scale_modifier,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const torch::Tensor& aabb,
    const bool with_distloss
) {
    const int N = positions.size(0);
    const int H = image_height;
    const int W = image_width;

    torch::TensorOptions float_opts = positions.options().dtype(torch::kFloat32);
    torch::TensorOptions byte_opts(torch::kByte);
    byte_opts = byte_opts.device(positions.device());

    torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
    torch::Tensor out_depth = torch::zeros({H, W}, float_opts);
    torch::Tensor out_alpha = torch::zeros({H, W}, float_opts);
    torch::Tensor out_distloss = with_distloss ? torch::zeros({H, W}, float_opts) : torch::empty({0}, float_opts);

    torch::Tensor geom_buffer = torch::empty({0}, byte_opts);
    torch::Tensor binning_buffer = torch::empty({0}, byte_opts);
    torch::Tensor img_buffer = torch::empty({0}, byte_opts);

    std::function<char*(size_t)> geom_func = resize_functional(geom_buffer);
    std::function<char*(size_t)> binning_func = resize_functional(binning_buffer);
    std::function<char*(size_t)> img_func = resize_functional(img_buffer);

    int rendered = 0;

    if (N > 0) {
        rendered = OctreeVoxelRasterizer::forward(
            geom_func,
            binning_func,
            img_func,
            N,
            background.contiguous().data_ptr<float>(),
            W, H,
            aabb.contiguous().data_ptr<float>(),
            positions.contiguous().data_ptr<float>(),
            colors.contiguous().data_ptr<float>(),
            densities.contiguous().data_ptr<float>(),
            depths.contiguous().data_ptr<uint8_t>(),
            scale_modifier,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            out_color.contiguous().data_ptr<float>(),
            out_depth.contiguous().data_ptr<float>(),
            out_alpha.contiguous().data_ptr<float>(),
            out_distloss.contiguous().data_ptr<float>()
        );
    }

    return std::make_tuple(
        rendered,
        out_color,
        out_depth,
        out_alpha,
        out_distloss,
        geom_buffer,
        binning_buffer,
        img_buffer
    );
}