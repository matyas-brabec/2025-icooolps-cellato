#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

#include "./simple.hpp"
#include "../../memory/standard_grid.hpp"
#include "../../memory/interface.hpp"
#include "../../evaluators/standard.hpp"
#include "../../core/ast.hpp"
#include "../traverser_utils.hpp"
#include "../cuda_utils.cuh"

namespace cellato::traversers::cuda::simple {

template <typename evaluator_t, typename grid_data_t, typename output_data_t>
__global__ void process_grid_kernel(
    grid_data_t input_data,
    output_data_t output_data,
    size_t width,
    size_t height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    cellato::memory::grids::point_in_grid state(input_data);

    state.properties.x_size = width;
    state.properties.y_size = height;
    state.position.x = x;
    state.position.y = y;

    auto result = evaluator_t::evaluate(state);
    save_to(output_data, state.idx(), result);
}

template <typename evaluator_type, typename grid_type>
template <_run_mode mode>
void traverser<evaluator_type, grid_type>::run_kernel(int steps) {

    auto current = &_input_grid_cuda;
    auto next = &_intermediate_grid_cuda;
    
    size_t width = current->x_size_physical();
    size_t height = current->y_size_physical();

    size_t width_threads = width - 2; // Exclude borders
    size_t height_threads = height - 2; // Exclude borders
    
    dim3 blockDim(_block_size_x, _block_size_y);
    dim3 gridDim(
        width_threads / blockDim.x,
        height_threads / blockDim.y
    );

    for (int step = 0; step < steps; ++step) {
        auto input_data = current->data();
        auto output_data = next->data();
        
        process_grid_kernel<evaluator_t><<<gridDim, blockDim>>>(
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

    _final_grid = current;
}

template <typename evaluator_type, typename grid_type>
auto  traverser<evaluator_type, grid_type>::fetch_result() -> grid_t {
    grid_t cpu_grid = _final_grid->to_cpu();

    _input_grid_cuda.free_cuda_memory();
    _intermediate_grid_cuda.free_cuda_memory();

    return cpu_grid;
}

} // namespace cellato::traversers::cuda::simple

#define SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS

#include "../../../src/game_of_life/cuda_instantiations.cuh"
#include "../../../src/fire/cuda_instantiations.cuh"
#include "../../../src/wire/cuda_instantiations.cuh"
#include "../../../src/greenberg/cuda_instantiations.cuh"

#undef SIMPLE_CUDA_TRAVERSER_INSTANTIATIONS
