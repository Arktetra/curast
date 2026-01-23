#include <cub/cub.cuh>
#include <cuda.h>
#include <cstdint>
#include <cooperative_groups.h>

#include "auxiliary.cuh"
#include "forward.h"
#include "data_structure.h"

namespace cg = cooperative_groups;

static uint32_t get_higher_msb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;

    while (step > 1) {
        step /= 2;
        
        if (n >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (n >> msb) msb++;
    return msb;
}

/**
 * Compute the morton code for a 3D point based on the camera position and the depth of the voxel.
 * 
 * @param pos Position of the point.
 * @param campos Camera position.
 * @param depth Depth of the voxel.
 */
static __device__ uint32_t compute_morton_code(float3 pos, float3 campos, uint8_t depth) {
    uint32_t mul = 1 << MAX_TREE_DEPTH;
	uint32_t xcode = (uint32_t)(pos.x * mul);
	uint32_t ycode = (uint32_t)(pos.y * mul);
	uint32_t zcode = (uint32_t)(pos.z * mul);
	uint32_t cxcode = (uint32_t)(campos.x * mul);
	uint32_t cycode = (uint32_t)(campos.y * mul);
	uint32_t czcode = (uint32_t)(campos.z * mul);
	uint32_t xflip = 0, yflip = 0, zflip = 0;
	bool done = false;
	for (int i = 1; i <= MAX_TREE_DEPTH && !done; i++)
	{
		xflip |= ((xcode >> (MAX_TREE_DEPTH - i + 1) << 1) < (cxcode >> (MAX_TREE_DEPTH - i))) ? (1 << (MAX_TREE_DEPTH - i)) : 0;
		yflip |= ((ycode >> (MAX_TREE_DEPTH - i + 1) << 1) < (cycode >> (MAX_TREE_DEPTH - i))) ? (1 << (MAX_TREE_DEPTH - i)) : 0;
		zflip |= ((zcode >> (MAX_TREE_DEPTH - i + 1) << 1) < (czcode >> (MAX_TREE_DEPTH - i))) ? (1 << (MAX_TREE_DEPTH - i)) : 0;
		done = i == depth;
	}
	xcode ^= xflip;
	ycode ^= yflip;
	zcode ^= zflip;
	return expand_bits(xcode) | (expand_bits(ycode) << 1) | (expand_bits(zcode) << 2);
}

static __global__ void preprocess(
    const int num_nodes,
    const float* positions,
    const uint8_t* tree_depths,
    const float scale_modifier,
    bool* clamped,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float3* cam_pos,
    const int width,
    const int height,
    const float tan_fovx,
    const float tan_fovy,
    const float focal_x,
    const float focal_y,
    const float* aabb,
    int4* bboxes,
    float* depths,
    float* rgb,
    const dim3 grid,
    uint32_t* tiles_touched,
    uint32_t* morton_codes
) {
    // Initialize bboxes and touched tiles to 0.
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_nodes)
        return;

    bboxes[idx] = {0, 0, 0, 0};
    tiles_touched[idx] = 0;

    // Perform near culling
    float3 p_orig = {
        positions[3 * idx] * aabb[0] + aabb[3],
        positions[3 * idx + 1] * aabb[1] + aabb[4],
        positions[3 * idx + 2] * aabb[2] + aabb[5]
    };
    float3 p_view;
    if (!in_frustum(p_orig, viewmatrix, projmatrix, p_view))
        return;

    // Project 8 vertices of the voxels to screen space to find
    // the bounding box of the projected points.
    float nsize = powf(2.0f, -(float)tree_depths[idx]) * scale_modifier;
	float3 scale = { aabb[3] * nsize, aabb[4] * nsize, aabb[5] * nsize };
	int4 bbox = get_bbox(p_orig, scale, projmatrix, width, height);
	uint2 rect_min, rect_max;
	get_rect(bbox, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    // Calculate view-dependent morton code for sorting.
    float3 pos = { positions[3 * idx], positions[3 * idx + 1], positions[3 * idx + 2] };
	float3 ncampos = {
		max(0.0f, min(1.0f, (cam_pos->x - aabb[0]) / aabb[3])),
		max(0.0f, min(1.0f, (cam_pos->y - aabb[1]) / aabb[4])),
		max(0.0f, min(1.0f, (cam_pos->z - aabb[2]) / aabb[5]))
	};
	uint32_t morton_code = compute_morton_code(pos, ncampos, tree_depths[idx]);

    // Store helper data for next steps.
    depths[idx] = p_view.z;
    bboxes[idx] = bbox;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    morton_codes[idx] = morton_code;
}

static __global__ void duplicate_with_keys(
    int N,
    const uint32_t* morton_codes,
    const uint32_t* offsets,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    int4* bboxes,
    dim3 grid
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx > N)
        return;

    if (bboxes[idx].w > 0) {
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;
        get_rect(bboxes[idx], rect_min, rect_max, grid);

        for (int y = rect_min.y; y < rect_max.y; y++) {
            for (int x = rect_min.x; x < rect_max.x; x++) {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= morton_codes[idx];
                keys_unsorted[off] = key;
                values_unsorted[off] = idx;
                off++;
            }
        }
    }

}

static __global__ void identify_tile_ranges(
    int N, 
    uint64_t* point_list_keys, 
    uint2* ranges
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx > N)
        return;

    uint64_t key = point_list_keys[idx];
    uint32_t curr_tile = key >> 32;
    if (idx == 0)
        ranges[curr_tile].x = 0;
    else {
        uint32_t prev_tile = point_list_keys[idx - 1] >> 32;
        if (curr_tile != prev_tile) {
            ranges[prev_tile].y = idx;
            ranges[curr_tile].x = idx;
        }
    }
    if (idx == N - 1)
        ranges[curr_tile].y = N - 1;
}

template<uint32_t CHANNELS>
static __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) render(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int W,
    const int H,
    const float* __restrict__ bg_color,
    const float3* cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ aabb,
    const float* __restrict__ positions,
    const float* __restrict__ features,
    const float* __restrict__ depths,
    const uint8_t* __restrict__ tree_depths,
    const float scale_modifier,
    const float* __restrict__ densities,
    float* __restrict__ final_T,
    float* __restrict__ final_wm_sum,
    uint32_t* __restrict__ n_contrib,
    float* __restrict__ out_color,
    float* __restrict__ out_depth,
    float* __restrict__ out_alpha,
    float* __restrict__ out_distloss
) {
    auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;

	// Get ray direction and origin for this pixel.
	float3 ray_dir = get_ray_dir(pix, W, H, tan_fovx, tan_fovy, viewmatrix);

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_xyz[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];
	__shared__ float collected_densities[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0;
	float wm_prefix = 0;
	float distloss = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-voxel data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = {
				positions[3 * coll_id] * aabb[3] + aabb[0],
				positions[3 * coll_id + 1] * aabb[4] + aabb[1],
				positions[3 * coll_id + 2] * aabb[5] + aabb[2]
			};
			float nsize = powf(2.0f, -(float)tree_depths[coll_id]) * scale_modifier;
			collected_scales[block.thread_rank()] = {aabb[3] * nsize, aabb[4] * nsize, aabb[5] * nsize};
			collected_densities[block.thread_rank()] = densities[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Get ray-voxel intersection
			float3 p = collected_xyz[j];
			float3 scale = collected_scales[j];
			float3 voxel_min = { p.x - 0.5f * scale.x, p.y - 0.5f * scale.y, p.z - 0.5f * scale.z };
			float3 voxel_max = { p.x + 0.5f * scale.x, p.y + 0.5f * scale.y, p.z + 0.5f * scale.z };
			float2 itsc = get_ray_voxel_intersection(*cam_pos, ray_dir, voxel_min, voxel_max);
			float itsc_dist = (itsc.y >= itsc.x) ? itsc.y - itsc.x : -1.0f;
			if (itsc_dist <= 0.0f)
				continue;

			// Volume rendering
			float alpha = min(1 - exp(-collected_densities[j] * itsc_dist), 0.999f);
			const float weight = alpha * T;

			// Accumulate color and depth
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
			D += depths[collected_id[j]] * weight;

			// Distortion loss
			// loss_bi := 2 * (wm * w_prefix - w * wm_prefix); loss_uni := 1.0f / 3.0f * (itsc_dist * w^2);
			if (out_distloss != nullptr)
			{
				float midpoint = 0.5f * (itsc.x + itsc.y);
				float wm = weight * midpoint;
				distloss += 2.0f * (wm * (1.0f - T) - weight * wm_prefix) + (1.0f / 3.0f) * itsc_dist * weight * weight;
				wm_prefix += wm;
			}

			T *= 1 - alpha;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// If we have accumulated enough, we can stop
			if (T < 0.001f)
				done = true;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
		out_alpha[pix_id] = 1.0f - T;
		if (out_distloss != nullptr) {
			out_distloss[pix_id] = distloss;
			final_wm_sum[pix_id] = wm_prefix;
		}
	}
}

int OctreeVoxelRasterizer::forward(
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
) {
    DEBUG_PRINT("Starting forward pass\n");
    DEBUG_PRINT("   - number of nodes: %d\n", num_nodes);
    DEBUG_PRINT("   - image size: %d x %d\n", width, height);

    // Create a parallel config.
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Allocate buffers for geometry and image state.
    DEBUG_PRINT("Allocating buffers\n");
    size_t buffer_size;
    char* buffer_ptr;

    buffer_size = required_size<GeometryState>(num_nodes);
    DEBUG_PRINT("   - geometry buffer size: %zu\n", buffer_size);
    buffer_ptr = geometry_buffer(buffer_size);
    GeometryState geom_state = GeometryState::from_chunk(buffer_ptr, buffer_size);

    buffer_size = required_size<ImageState>(width * height);
    DEBUG_PRINT("   - image buffer size: %zu\n", buffer_size);
    buffer_ptr = image_buffer(buffer_size);
    ImageState image_state = ImageState::from_chunk(buffer_ptr, buffer_size);

    const float focal_x = height / (2.f * tan_fovx);
    const float focal_y = width / (2.f * tan_fovy);

    // Run preprocessing kernels
    DEBUG_PRINT("Calling preprocess kernel\n");
    CHECK_CUDA(
        preprocess<<<(num_nodes + 255) / 256, 256>>>(
            num_nodes,
            positions,
            depths,
            scale_modifier,
            geom_state.clamped,
            colors_precomp,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            tan_fovx,
            tan_fovy,
            focal_x,
            focal_y,
            aabb,
            geom_state.bboxes,
            geom_state.depths,
            geom_state.rgb,
            grid,
            geom_state.tiles_touched,
            geom_state.morton_codes
        )
    );

    // Compute prefix sum over the full list of touched tile counts by voxels
    CHECK_CUDA(
        cub::DeviceScan::InclusiveSum(
            geom_state.scanning_space,
            geom_state.scan_size,
            geom_state.tiles_touched,
            geom_state.point_offsets,
            num_nodes
        )
    );

    // Retrieve total number of voxel instances to launch.
    int num_rendered;
    CHECK_CUDA(
        cudaMemcpy(&num_rendered, geom_state.point_offsets + num_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost)
    );

    // Allocate buffer for binning state.
    DEBUG_PRINT("Allocating binning buffer\n");
    DEBUG_PRINT("   - number of rendered nodes: %d\n", num_rendered);
    buffer_size = required_size<BinningState>(num_rendered);
    DEBUG_PRINT("   - binning buffer size: %zu\n", buffer_size);
    buffer_ptr = binning_buffer(buffer_size);
    BinningState binning_state = BinningState::from_chunk(buffer_ptr, num_rendered);

    // Create [tile | depth] key and corresponding indices for each 
    // primitive to be rendered.
    DEBUG_PRINT("Calling duplicate_with_keys kernel\n");
    CHECK_CUDA(
        duplicate_with_keys<<<(num_nodes + 255) / 256, 256>>>(
            num_nodes,
            geom_state.morton_codes,
            geom_state.point_offsets,
            binning_state.point_list_keys_unsorted,
            binning_state.point_list_unsorted,
            geom_state.bboxes,
            grid
        )
    );

    // Sort the complete list of voxel indices by keys.
    int bit = get_higher_msb(grid.x * grid.y);
    CHECK_CUDA(
        cub::DeviceRadixSort::SortPairs(
            binning_state.list_sorting_space,
            binning_state.sorting_size,
            binning_state.point_list_keys_unsorted,
            binning_state.point_list_keys,
            binning_state.point_list_unsorted,
            binning_state.point_list,
            num_rendered,
            0,
            32 + bit
        )
    );

    // Identify start and end of per-tile workloads in sorted list.
    CHECK_CUDA(
        cudaMemset(image_state.ranges, 0, grid.x * grid.y * sizeof(uint2));
    )
    CHECK_CUDA(
        identify_tile_ranges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binning_state.point_list_keys,
            image_state.ranges
        )
    );

    // Let each tile blend its range of voxels independently in parallel.
    const float* color_ptr = colors_precomp != nullptr ? colors_precomp : geom_state.rgb;
    DEBUG_PRINT("Calling render kernel\n");
    CHECK_CUDA(
        render<NUM_CHANNELS><<<grid, block>>>(
            image_state.ranges,
            binning_state.point_list,
            width, height,
            background,
            cam_pos,
            tan_fovx,
            tan_fovy,
            viewmatrix,
            aabb,
            positions,
            color_ptr,
            geom_state.depths,
            depths,
            scale_modifier,
            densities,
            image_state.accum_alpha,
            image_state.wm_sum,
            image_state.n_contrib,
            out_color,
            out_depth,
            out_alpha,
            out_distloss
        )
    );

    return num_rendered;
}