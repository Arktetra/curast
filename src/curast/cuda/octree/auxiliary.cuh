#pragma once
#include <cstdint>
#include "config.h"

__forceinline__ __device__ uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__forceinline__ __device__ void get_rect(const int4 bbox, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((bbox.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((bbox.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((bbox.z + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((bbox.w + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transform_point_4x3(const float3& p, const float* matrix) {
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};

	return transformed;
}

__forceinline__ __device__ float4 transform_point_4x4(const float3& p, const float* matrix) {
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    
    return transformed;
}

__forceinline__ __device__ bool in_frustum(
    const float3& p_orig,
    const float* viewmatrix,
    const float* projmatrix,
    float3& p_view
) {
    float4 p_hom = transform_point_4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
    p_view = transform_point_4x3(p_orig, viewmatrix);

    if (p_view.z <= 0.2f) {
        return false;
    }
    return true;
}

__forceinline__ __device__ int2 project(const float3& p, const float* matrix, const int& width, const int& height)
{
	float3 p_hom = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	float p_w = 1.0f / (p_hom.z + 0.0000001f);
	return { (int)((p_hom.x * p_w + 1.0f) * 0.5f * width), (int)((p_hom.y * p_w + 1.0f) * 0.5f * height) };
}

#define GET_BBOX_FIRST(A, B, C) \
vertex.x = point.x A half_scale.x; \
vertex.y = point.y B half_scale.y; \
vertex.z = point.z C half_scale.z; \
p_screen = project(vertex, projmatrix, width, height); \
bbox.x = p_screen.x; \
bbox.y = p_screen.y; \
bbox.z = p_screen.x + 1; \
bbox.w = p_screen.y + 1;

#define GET_BBOX_OTHER(A, B, C) \
vertex.x = point.x A half_scale.x; \
vertex.y = point.y B half_scale.y; \
vertex.z = point.z C half_scale.z; \
p_screen = project(vertex, projmatrix, width, height); \
bbox.x = min(bbox.x, p_screen.x); \
bbox.y = min(bbox.y, p_screen.y); \
bbox.z = max(bbox.z, p_screen.x + 1); \
bbox.w = max(bbox.w, p_screen.y + 1);

__forceinline__ __device__ int4 get_bbox(
	const float3& point,
	const float3& scale,
	const float* projmatrix,
	const int& width,
	const int& height
) {
	float3 half_scale = { scale.x * 0.5f, scale.y * 0.5f, scale.z * 0.5f };
	float3 vertex;
	int2 p_screen;
	int4 bbox;

	GET_BBOX_FIRST(-, -, -);
	GET_BBOX_OTHER(+, -, -);
	GET_BBOX_OTHER(-, +, -);
	GET_BBOX_OTHER(+, +, -);
	GET_BBOX_OTHER(-, -, +);
	GET_BBOX_OTHER(+, -, +);
	GET_BBOX_OTHER(-, +, +);
	GET_BBOX_OTHER(+, +, +);
	
	bbox.x = max(0, bbox.x);
	bbox.y = max(0, bbox.y);
	bbox.z = min(width, bbox.z);
	bbox.w = min(height, bbox.w);
	if (bbox.x >= bbox.z || bbox.y >= bbox.w)	// bbox is empty
		return { 0, 0, 0, 0 };
	return bbox;
}

__forceinline__ __device__ float2 get_ray_voxel_intersection(
	const float3& ray_origin,
	const float3& ray_direction,
	const float3& voxel_min,
	const float3& voxel_max
) {
	// Careful with the division by zero
	float3 inv_direction;
	inv_direction.x = ray_direction.x == 0.0f ? 1e10f : 1.0f / ray_direction.x;
	inv_direction.y = ray_direction.y == 0.0f ? 1e10f : 1.0f / ray_direction.y;
	inv_direction.z = ray_direction.z == 0.0f ? 1e10f : 1.0f / ray_direction.z;
	float3 t0 = {
		(voxel_min.x - ray_origin.x) * inv_direction.x,
		(voxel_min.y - ray_origin.y) * inv_direction.y,
		(voxel_min.z - ray_origin.z) * inv_direction.z
	};
	float3 t1 = {
		(voxel_max.x - ray_origin.x) * inv_direction.x,
		(voxel_max.y - ray_origin.y) * inv_direction.y,
		(voxel_max.z - ray_origin.z) * inv_direction.z
	};
	float3 tmin = {
		min(t0.x, t1.x),
		min(t0.y, t1.y),
		min(t0.z, t1.z)
	};
	float3 tmax = {
		max(t0.x, t1.x),
		max(t0.y, t1.y),
		max(t0.z, t1.z)
	};
	float tmin_max = max(tmin.x, max(tmin.y, tmin.z));
	float tmax_min = min(tmax.x, min(tmax.y, tmax.z));
	return { tmin_max, tmax_min };
}

__forceinline__ __device__ float3 normalize(const float3& v)
{
	float inv_norm = 1.0f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x * inv_norm, v.y * inv_norm, v.z * inv_norm };
}

__forceinline__ __device__ float3 get_ray_dir(
	const uint2& pix,
	const int& width,
	const int& height,
	const float& tan_fovx,
	const float& tan_fovy,
	const float* viewmatrix
) {
	float x = (2.0f * (pix.x + 0.5f) / width - 1.0f) * tan_fovx;
	float y = (2.0f * (pix.y + 0.5f) / height - 1.0f) * tan_fovy;
	float3 ray_dir = {
		viewmatrix[0] * x + viewmatrix[1] * y + viewmatrix[2],
		viewmatrix[4] * x + viewmatrix[5] * y + viewmatrix[6],
		viewmatrix[8] * x + viewmatrix[9] * y + viewmatrix[10]
	};
	return normalize(ray_dir);
}


#ifdef DEBUG
    #define CUDA_CHECK(expr_to_check) do {            \
        cudaError_t result  = expr_to_check;          \
        if(result != cudaSuccess)                     \
        {                                             \
            fprintf(stderr,                           \
                    "CUDA Runtime Error: %s:%i:%d = %s\n", \
                    __FILE__,                         \
                    __LINE__,                         \
                    result,\
                    cudaGetErrorString(result));      \
        }                                             \
    } while(0)
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
    #define CHECK_CUDA(...) __VA_ARGS__
    #define DEBUG_PRINT(...)
#endif