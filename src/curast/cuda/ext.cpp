#include <torch/extension.h>
#include "octree/api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_octree_voxels", &OctreeVoxelRasterizer::rasterize_cuda);
}