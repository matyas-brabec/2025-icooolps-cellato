#ifndef FOREST_FIRE_DATA_INIT_HPP
#define FOREST_FIRE_DATA_INIT_HPP

#include <vector>
#include <random>

#include "experiments/run_params.hpp"

namespace fire {

struct fire_random_init {
    static std::vector<fire_cell_state> init(cellib::run::run_params& params) {
        std::vector<fire_cell_state> initial_state(params.x_size * params.y_size);
        
        // Probabilities for each cell state
        std::vector<std::tuple<fire_cell_state, double>> probabilities = {
            {fire_cell_state::empty, 0.20},   // 20% empty cells
            {fire_cell_state::tree, 0.79},    // 79% trees
            {fire_cell_state::fire, 0.01},    // 1% fire (ignition points)
            {fire_cell_state::ash, 0.00}      // 0% ash initially
        };
        
        // Generate random grid using utility
        cellib::memory::grids::utils::generate_random_grid(
            initial_state,
            params.y_size, params.x_size,
            probabilities
        );
        
        return initial_state;
    }
};

}

#endif // FOREST_FIRE_DATA_INIT_HPP
