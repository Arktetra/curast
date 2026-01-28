#pragma once
#include <torch/extension.h>

namespace OctreeVoxelRasterizer {

    /**
     * Rasterize a sparse voxel octree with CUDA backend.
     * 
     * @param background        Tensor of shape (3, ) containing the background color.
     * @param positions         Tensor of shape (N, 3) containing the positions of octree nodes in [0, 1]^3.
     * @param colors            Tensor of shape (N, 3) containing the colors.
     * @param densities         Tensor of shape (N, 1) containing the densities.
     * @param depths            Tensor of shape (N, 1) containing the depths of the cotree nodes.
     * @param scale_modifier    Float containing the scale modifier.
     * @param viewmatrix        Tensor of shape (4, 4) containing the view matrix.
     * @param projmatrix        Tensor of shape (4, 4) containing the projection matrix.
     * @param tan_fovx          Float containing the tangent of the horizontal field of view.
     * @param tan_fovy          Float containing the tangent of the vertical field of view.
     * @param image_height      Integer containing the image height.
     * @param image_width       Integer containing the image width.
     * @param campos            Tensor of shape (3, ) containing the camera position.
     * @param aabb              Tensor of shape (6, ) containing the axis-aligned bounding box.
     * @param with_distloss     Boolean containing whether to return the distance loss.
     * 
     * @return A tuple containing:
     *  - Integer containing the number of points being rasterized.
     *  - Tensor of shape (3, H, W) containing the output color.
     *  - Tensor of shape (H, W) containing the output depth.
     *  - Tensor of shape (H, W) containing the output alpha.
     *  - Tensor of shape (H, W) containing the distance loss.
     *  - Tensor used for geometry buffer.
     *  - Tensor used for binning buffer.
     *  - Tensor used for image buffer.
     */
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_cuda(
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
    );
}