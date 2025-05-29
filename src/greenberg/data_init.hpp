#ifndef GREENBERG_HASTINGS_DATA_INIT_HPP
#define GREENBERG_HASTINGS_DATA_INIT_HPP

#include <vector>
#include <random>
#include "memory/grid_utils.hpp"
#include "experiments/run_params.hpp"
#include "./algorithm.hpp"

namespace greenberg {

struct ghm_random_init {
    static std::vector<ghm_cell_state> init(cellato::run::run_params& params) {
        std::vector<ghm_cell_state> initial_state(params.x_size * params.y_size);
        
        // Create a more sustainable distribution of states:
        // - More excited cells (15% instead of 12%)
        // - Some cells already in refractory states to create richer patterns
        std::vector<std::tuple<ghm_cell_state, double>> probabilities = {
            {ghm_cell_state::quiescent, 0.70},    // 70% quiescent
            {ghm_cell_state::excited, 0.15},      // 15% excited
            {ghm_cell_state::refractory_1, 0.05}, // 5% refractory_1
            {ghm_cell_state::refractory_2, 0.03}, // 3% refractory_2
            {ghm_cell_state::refractory_3, 0.03}, // 3% refractory_3
            {ghm_cell_state::refractory_4, 0.02}, // 2% refractory_4
            {ghm_cell_state::refractory_5, 0.01}, // 1% refractory_5
            {ghm_cell_state::refractory_6, 0.01}  // 1% refractory_6
        };
        
        // Generate random grid using utility
        cellato::memory::grids::utils::generate_random_grid(
            initial_state,
            params.y_size, params.x_size,
            probabilities,
            params.seed
        );
        
        return initial_state;
    }
};

}

#endif // GREENBERG_HASTINGS_DATA_INIT_HPP
