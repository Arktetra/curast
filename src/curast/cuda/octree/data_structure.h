#pragma once

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include "config.h"

namespace OctreeVoxelRasterizer {

    template<typename T>
    static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment) {
        size_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        chunk = reinterpret_cast<char*>(ptr + count);
    }

    template<typename T>
    size_t required_size(size_t N) {
        char* size = nullptr;
        T::from_chunk(size, N);
        return (reinterpret_cast<uintptr_t>(size) + MEM_ALIGNMENT - 1) & ~(MEM_ALIGNMENT - 1);
    }

    struct GeometryState {
        size_t scan_size;
        float* depths;
        uint32_t* morton_codes;
        char* scanning_space;
        bool* clamped;
        int4* bboxes;
        float* rgb;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        /**
         * Initiate `GeometryState` from a chunk.
         * @param chunk     a pointer to the chunk.
         * @param N         number of elements.
         * @return a `GeometryState` object.
         */
        static GeometryState from_chunk(char*& chunk, size_t N);
    };

    struct ImageState {
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;
        float* wm_sum;

        /**
         * Initiate `ImageState` from a chunk.
         * @param chunk     a pointer to the chunk.
         * @param N         number of elements.
         * @return an `ImageState` object.
         */
        static ImageState from_chunk(char*& chunk, size_t N);
    };

    struct BinningState {
        size_t sorting_size;
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;
        char* list_sorting_space;

        /**
         * Initiate `BinningState` from a chunk.
         * @param chunk     a pointer to the chunk.
         * @param N         number of elements.
         * @return a `BinningState` object.
         */
        static BinningState from_chunk(char*& chunk, size_t N);
    };
}