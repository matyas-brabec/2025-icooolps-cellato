#ifndef CELLIB_TRAVERSERS_CPU_SIMPLE_HPP
#define CELLIB_TRAVERSERS_CPU_SIMPLE_HPP

#include <iostream>
#include <thread>
#include <chrono>
#include <utility>
#include <functional>

#include "../../memory/interface.hpp"
#include "../traverser_utils.hpp"
#include "../../experiments/run_params.hpp"

namespace cellib::traversers::cpu::simple {

using namespace cellib::traversers::utils;

template <
    typename evaluator_type,
    typename grid_type >
class traverser {
    using evaluator_t = evaluator_type;
    using grid_t = grid_type;

    using cell_t = typename grid_t::store_type;

  public:

    void init(grid_t grid, 
              const cellib::run::run_params& params) {
        (void)params; // Unused parameter

        _input_grid = std::move(grid);
        _intermediate_grid = _input_grid;
        _final_grid = &_intermediate_grid;
    }

    struct no_callback {};

    template <typename callback = no_callback>
    void run(int steps, callback&& callback_func = no_callback{}) {
        
        auto current = &_input_grid;
        auto next = &_intermediate_grid;

        auto state = cellib::memory::grids::point_in_grid(current->data());

        state.properties.x_size = _input_grid.x_size_physical();
        state.properties.y_size = _input_grid.y_size_physical();

        if constexpr (!std::is_same_v<callback, no_callback>) {
            callback_func(0, _input_grid);
        }

        if constexpr (!std::is_same_v<callback, no_callback>) {
            callback_func(0, *current);
        }

        for (int step = 0; step < steps; ++step) {

            state.grid = current->data();
            auto next_data = next->data();

            // Process cells (skip border)
            for (std::size_t y = 1; y < state.properties.y_size - 1; ++y) {
                for (std::size_t x = 1; x < state.properties.x_size - 1; ++x) {

                    state.position.x = x;
                    state.position.y = y;

                    auto result = evaluator_t::evaluate(state);
                    save_to(next_data, state.idx(), result);
                }
            }

            // Call the callback function if provided
            if constexpr (!std::is_same_v<callback, no_callback>) {
                callback_func(step + 1, *next);
            }

            std::swap(current, next);
        }

        _final_grid = current;
    }

    grid_t fetch_result() const {
        return std::move(*_final_grid);
    }

    private:

    grid_t _input_grid, _intermediate_grid;
    grid_t* _final_grid;
};

}

#endif // CELLIB_TRAVERSERS_CPU_SIMPLE_HPP