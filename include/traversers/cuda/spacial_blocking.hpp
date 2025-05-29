#ifndef CELLIB_TRAVERSERS_CUDA_SPACIAL_BLOCKING_HPP
#define CELLIB_TRAVERSERS_CUDA_SPACIAL_BLOCKING_HPP

#include <iostream>
#include <utility>
#include <functional>
#include <cuda_runtime.h>
#include <memory>

#include "../../memory/interface.hpp"
#include "../../experiments/run_params.hpp"
#include "../traverser_utils.hpp"

namespace cellib::traversers::cuda::spacial_blocking {

using namespace cellib::traversers::utils;

enum class _run_mode {
    QUIET,
    VERBOSE,
};

template <
    typename evaluator_type,
    typename grid_type,
    int Y_TILE_SIZE,
    int X_TILE_SIZE>
class traverser {
    using evaluator_t = evaluator_type;
    using grid_t = grid_type;
    using cuda_grid_t = typename std::invoke_result<decltype(&grid_t::to_cuda), grid_t>::type;
    using cell_t = typename grid_t::store_type;

  public:
    traverser() : _final_grid(nullptr) {}

    void init(grid_t grid,
              const cellib::run::run_params& params) {
        _block_size_x = params.cuda_block_size_x;
        _block_size_y = params.cuda_block_size_y;

        _input_grid = std::move(grid);
        
        _input_grid_cuda = _input_grid.to_cuda();
        _intermediate_grid_cuda = _input_grid.to_cuda();
        
        _final_grid = &_input_grid_cuda;
    }

    struct no_callback {};

    template <typename callback = no_callback>
    void run(int steps, callback&& callback_func = no_callback{}) {
        
        if constexpr (!std::is_same_v<callback, no_callback>) {
            _callback_func = std::make_unique<_lambda_wrapper<callback>>(std::forward<callback>(callback_func));
            run_kernel<_run_mode::VERBOSE>(steps);

        } else {
            run_kernel<_run_mode::QUIET>(steps);
        }
    }
    
    grid_t fetch_result();
    
private:
    template <_run_mode mode>
    void run_kernel(int steps);
    
    grid_t _input_grid;
    cuda_grid_t _input_grid_cuda;
    cuda_grid_t _intermediate_grid_cuda;
    cuda_grid_t* _final_grid;

    int _block_size_x = X_TILE_SIZE;
    int _block_size_y = Y_TILE_SIZE;

    struct _call_back_obj {
        virtual void call(int iteration, grid_t& grid) = 0;
    };

    template <typename func>
    struct _lambda_wrapper : public _call_back_obj {
        func f;
        _lambda_wrapper(func f) : f(f) {}
        void call(int iteration, grid_t& grid) override { f(iteration, grid); }
    };

    std::unique_ptr<_call_back_obj> _callback_func;

    void call_callback(int iteration, cuda_grid_t* grid) {
        if (_callback_func) {
            auto on_host_grid = grid->to_cpu();
            _callback_func->call(iteration, on_host_grid);
        }
    }
};

} // namespace cellib::traversers::cuda::spacial_blocking

#endif // CELLIB_TRAVERSERS_CUDA_SPACIAL_BLOCKING_HPP
