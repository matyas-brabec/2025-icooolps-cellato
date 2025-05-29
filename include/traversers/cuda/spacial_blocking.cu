#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

#include "./spacial_blocking.hpp"
#include "../../memory/standard_grid.hpp"
#include "../../memory/interface.hpp"
#include "../../evaluators/standard.hpp"
#include "../../core/ast.hpp"
#include "../traverser_utils.hpp"
#include "../cuda_utils.cuh"

namespace cellib::traversers::cuda::spacial_blocking {

template <typename evaluator_t, typename grid_data_t, typename output_data_t, int Y_TILE_SIZE, int X_TILE_SIZE>
__global__ void process_grid_kernel(
    grid_data_t input_data,
    output_data_t output_data,
    size_t width,
    size_t height
) {
    // Calculate base coordinates for this thread's tile
    int base_x = blockIdx.x * blockDim.x + threadIdx.x;
    int base_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Each thread handles a Y_TILE_SIZE x X_TILE_SIZE tile of cells
    // Process each cell in the tile
    for (int tile_y = 0; tile_y < Y_TILE_SIZE; tile_y++) {
        int y = base_y * Y_TILE_SIZE + tile_y;
        
        // Skip if y is out of bounds or on the border
        if (y <= 0 || y >= height - 1) continue;
        
        for (int tile_x = 0; tile_x < X_TILE_SIZE; tile_x++) {
            int x = base_x * X_TILE_SIZE + tile_x;
            
            // Skip if x is out of bounds or on the border
            if (x <= 0 || x >= width - 1) continue;
            
            // Process this cell
            cellib::memory::grids::point_in_grid state(input_data);
            state.properties.x_size = width;
            state.properties.y_size = height;
            state.position.x = x;
            state.position.y = y;
            
            auto result = evaluator_t::evaluate(state);
            save_to(output_data, state.idx(), result);
        }
    }
}

template <typename evaluator_type, typename grid_type, int Y_TILE_SIZE, int X_TILE_SIZE>
template <_run_mode mode>
void traverser<evaluator_type, grid_type, Y_TILE_SIZE, X_TILE_SIZE>::run_kernel(int steps) {
    auto current = &_input_grid_cuda;
    auto next = &_intermediate_grid_cuda;
    
    size_t width = current->x_size_physical();
    size_t height = current->y_size_physical();
    
    // Calculate block and grid dimensions based on tile sizes
    // Each thread handles a Y_TILE_SIZE x X_TILE_SIZE tile
    // So we need fewer threads than with the simple traverser
    dim3 blockDim(_block_size_x, _block_size_y);
    dim3 gridDim(
        (width + blockDim.x * X_TILE_SIZE - 1) / (blockDim.x * X_TILE_SIZE),
        (height + blockDim.y * Y_TILE_SIZE - 1) / (blockDim.y * Y_TILE_SIZE)
    );

    for (int step = 0; step < steps; ++step) {
        auto input_data = current->data();
        auto output_data = next->data();
        
        process_grid_kernel<evaluator_t, decltype(input_data), decltype(output_data), Y_TILE_SIZE, X_TILE_SIZE><<<gridDim, blockDim>>>(
            input_data,
            output_data,
            width,
            height
        );
        
        if constexpr (mode == _run_mode::VERBOSE) {
            call_callback(step, current);
        }

        std::swap(current, next);
    }
    
    CUCH(cudaDeviceSynchronize());

    if (steps % 2 == 1) {
        _final_grid = next;
    } else {
        _final_grid = current;
    }
}

template <typename evaluator_type, typename grid_type, int Y_TILE_SIZE, int X_TILE_SIZE>
auto traverser<evaluator_type, grid_type, Y_TILE_SIZE, X_TILE_SIZE>::fetch_result() -> grid_t {
    grid_t cpu_grid = _final_grid->to_cpu();

    _input_grid_cuda.free_cuda_memory();
    _intermediate_grid_cuda.free_cuda_memory();

    return cpu_grid;
}

} // namespace cellib::traversers::cuda::spacial_blocking


#define SPACIAL_BLOCKING_CUDA_TRAVERSER_INSTANTIATIONS

// Include necessary instantiations for all the cellular automata
#include "../../../src/game_of_life/cuda_instantiations.cuh"
#include "../../../src/fire/cuda_instantiations.cuh"
#include "../../../src/wire/cuda_instantiations.cuh"
#include "../../../src/greenberg/cuda_instantiations.cuh"

#undef SPACIAL_BLOCKING_CUDA_TRAVERSER_INSTANTIATIONS
