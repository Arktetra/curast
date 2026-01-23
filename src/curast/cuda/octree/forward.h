#include <cstdint>
#include <functional>

static uint32_t get_higher_msb(uint32_t n);

namespace OctreeVoxelRasterizer {

    /**
     * Forward pass of the voxel rasterization with CUDA backend.
     * 
     * @param geometry_buffer       function to allocate the geometry buffer,
     * @param binning_buffer        function to allocate the binning buffer.
     * @param image_buffer          function to allocate the image buffer.
     * @param num_nodes             integer containing the number of octree leaves.
     * @param background            pointer to the background color.
     * @param width                 integer containing the image width.
     * @param height                integer containing the image height.
     * @param aabb                  pointer to the axis-aligned bounding box.
     * @param positions             pointer to the positions of octree nodes.
     * @param colors_precomp        pointer to the precomputed colors.
     * @param densities             pointer to the densities.
     * @param depths                pointer to the depths of the octree nodes.
     * @param scale_modifier        float containing the scale modifier.
     * @param viewmatrix            pointer to the view matrix.
     * @param projmatrix            pointer to the projection matrix.
     * @param cam_pos               pointer to the camera position.
     * @param tan_fovx              float containing the tangent of the horizontal field of view.
     * @param tan_fovy              float containing the tangent of the vertical field of view.
     * @param out_color             pointer to the output color.
     * @param out_depth             pointer to the output depth.
     * @param out_alpha             pointer to the output alpha.
     * @param out_distloss          pointer to the output distance loss.
     * 
     * @return integer containing the number of nodes being rasterized.
     */
    int forward(
        std::function<char*(size_t)> geometry_buffer,
        std::function<char*(size_t)> binning_buffer,
        std::function<char*(size_t)> image_buffer,
        const int num_nodes,
        const float* background,
        const int width,
        const int height,
        const float* aabb,
        const float* positions,
        const float* colors_precomp,
        const float* densities,
        const uint8_t* depths,
        const float scale_modifier,
        const float* viewmatrix,
        const float* projmatrix,
        const float3* cam_pos,
        const float tan_fovx,
        const float tan_fovy,
        float* out_color,
        float* out_depth,
        float* out_alpha,
        float* out_distloss
    );
}