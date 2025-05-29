#include "./reference_implementation.hpp"
#include <cuda_runtime.h>
#include "traversers/cuda_utils.cuh"

namespace game_of_life::reference {

// CUDA kernel for Game of Life (single step)
__global__ void gol_kernel(const gol_cell_state* current, gol_cell_state* next, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
        
    const int idx = y * width + x;
    
    // Count live neighbors using explicit indexing (Moore neighborhood)
    int live_neighbors = 
        (current[(y - 1) * width + (x - 1)] == gol_cell_state::alive) + // Top-left
        (current[(y - 1) * width +  x     ] == gol_cell_state::alive) + // Top
        (current[(y - 1) * width + (x + 1)] == gol_cell_state::alive) + // Top-right
        (current[ y      * width + (x - 1)] == gol_cell_state::alive) + // Left
        (current[ y      * width + (x + 1)] == gol_cell_state::alive) + // Right
        (current[(y + 1) * width + (x - 1)] == gol_cell_state::alive) + // Bottom-left
        (current[(y + 1) * width +  x     ] == gol_cell_state::alive) + // Bottom
        (current[(y + 1) * width + (x + 1)] == gol_cell_state::alive);  // Bottom-right
    
    // Apply Game of Life rules
    gol_cell_state cell_state = current[idx];
    
    if (cell_state == gol_cell_state::alive) {
        // Live cell with fewer than 2 or more than 3 live neighbors dies
        if (live_neighbors < 2 || live_neighbors > 3) {
            next[idx] = gol_cell_state::dead;
        } else {
            // Live cell with 2 or 3 live neighbors stays alive
            next[idx] = gol_cell_state::alive;
        }
    } else {
        // Dead cell with exactly 3 live neighbors becomes alive
        if (live_neighbors == 3) {
            next[idx] = gol_cell_state::alive;
        } else {
            // Dead cell stays dead
            next[idx] = gol_cell_state::dead;
        }
    }
}

void runner::run_kernel(int steps) {
    if (!d_current || !d_next) {
        init_cuda();
    }
    
    // Set up grid and block dimensions
    dim3 block_size(_block_size_x, _block_size_y);

    auto _x_size_threads = _x_size - 2; // Exclude borders
    auto _y_size_threads = _y_size - 2; // Exclude borders

    dim3 grid_dim((_x_size_threads + block_size.x - 1) / block_size.x, 
                 (_y_size_threads + block_size.y - 1) / block_size.y);
    
    // Run steps iterations
    for (int i = 0; i < steps; i++) {
        // Launch kernel for one step
        gol_kernel<<<grid_dim, block_size>>>(d_current, d_next, _x_size, _y_size);

        // Swap pointers for next iteration
        gol_cell_state* temp = d_current;
        d_current = d_next;
        d_next = temp;
    }
    
    CUCH(cudaDeviceSynchronize());
}

} // namespace game_of_life::reference
