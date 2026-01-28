#include <cub/cub.cuh>
#include "data_structure.h"
#include "config.h"

OctreeVoxelRasterizer::GeometryState OctreeVoxelRasterizer::GeometryState::from_chunk(char*& chunk, size_t N) {
    GeometryState geom;
    obtain(chunk, geom.depths, N, MEM_ALIGNMENT);
    obtain(chunk, geom.morton_codes, N, MEM_ALIGNMENT);
    obtain(chunk, geom.clamped, N * 3, MEM_ALIGNMENT);
    obtain(chunk, geom.bboxes, N, MEM_ALIGNMENT);
    obtain(chunk, geom.rgb, N * 3, MEM_ALIGNMENT);
    obtain(chunk, geom.tiles_touched, N, MEM_ALIGNMENT);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, N);
    obtain(chunk, geom.scanning_space, geom.scan_size, MEM_ALIGNMENT);
    obtain(chunk, geom.point_offsets, N, MEM_ALIGNMENT);

    return geom;
}

OctreeVoxelRasterizer::ImageState OctreeVoxelRasterizer::ImageState::from_chunk(char*& chunk, size_t N) {
    ImageState img;
    obtain(chunk, img.accum_alpha, N, MEM_ALIGNMENT);
    obtain(chunk, img.wm_sum, N, MEM_ALIGNMENT);
    obtain(chunk, img.n_contrib, N, MEM_ALIGNMENT);
    obtain(chunk, img.ranges, N, MEM_ALIGNMENT);

    return img;
}

OctreeVoxelRasterizer::BinningState OctreeVoxelRasterizer::BinningState::from_chunk(char*& chunk, size_t N) {
    BinningState binning;
    obtain(chunk, binning.point_list, N, MEM_ALIGNMENT);
    obtain(chunk, binning.point_list_unsorted, N, MEM_ALIGNMENT);
    obtain(chunk, binning.point_list_keys, N, MEM_ALIGNMENT);
    obtain(chunk, binning.point_list_keys_unsorted, N, MEM_ALIGNMENT);
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        binning.sorting_size,
        binning.point_list_keys_unsorted,
        binning.point_list_keys,
        binning.point_list_unsorted,
        binning.point_list,
        N
    );
    obtain(chunk, binning.list_sorting_space, N, MEM_ALIGNMENT);
    
    return binning;
}
