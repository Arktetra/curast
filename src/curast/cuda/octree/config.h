#pragma once

#define NUM_CHANNELS 3      // Default to R, G, B
#define BLOCK_X 8
#define BLOCK_Y 8
#define MEM_ALIGNMENT 128
#define MAX_TREE_DEPTH 10
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)