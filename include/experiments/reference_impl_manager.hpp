#ifndef REFERENCE_IMPL_MANAGER_HPP
#define REFERENCE_IMPL_MANAGER_HPP

#include <vector>
#include <chrono>
#include <iostream>
#include <string>

#include "./run_params.hpp"
#include "./experiment_report.hpp"
#include "memory/standard_grid.hpp"

namespace cellato::run {

template <typename RunnerT, typename CellStateT>
class reference_impl_manager {
public:
    using runner_t = RunnerT;
    using cell_state_t = CellStateT;
    using standard_grid_t = cellato::memory::grids::standard::grid<cell_state_t>;
    
    // Constant margin size for all reference implementations
    constexpr static int margin = 1;
    
    reference_impl_manager() = default;
    
    experiment_report run_experiment(const run_params& params, const std::vector<cell_state_t>& initial_state) {
        experiment_report report;
        report.params = params;
        
        for (int i = 0; i < params.warmup_rounds + params.rounds; ++i) {
            auto [duration, checksum] = run_round(i, params, initial_state);
            
            if (i >= params.warmup_rounds) {
                report.execution_times_ms.push_back(duration);
                report.checksums.push_back(checksum);
            }
        }
        
        return report;
    }
    
private:
    std::tuple<double, std::string> run_round(int round, const run_params& params, 
                                              const std::vector<cell_state_t>& initial_state) {
        if (round < params.warmup_rounds) {
            std::cerr << "\nWarmup round: " << round << "\n";
        }
        else {
            std::cerr << "\nRound: " << round - params.warmup_rounds << "\n";
        }
        
        // Create standard grid and add margins
        standard_grid_t initial_grid(params.x_size, params.y_size);
        std::copy(initial_state.begin(), initial_state.end(), initial_grid.data());
        
        auto padded_grid = initial_grid.template with_empty_margins<margin, margin>();
        
        run_params padded_params = params;
        padded_params.x_size += 2 * margin;
        padded_params.y_size += 2 * margin;
        
        // Initialize the runner with padded grid
        runner_t runner;
        runner.init(padded_grid.data(), padded_params);
        
        if (params.device == "CUDA") {
            runner.init_cuda();
        }
        
        // Run the appropriate version based on device
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (params.device == "CUDA") {
            runner.run_on_cuda(params.steps);
        } else {
            runner.run(params.steps);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
        
        // Fetch results and remove margins
        auto result = runner.fetch_result();
        
        // Create a new standard grid from the result
        standard_grid_t result_grid(padded_grid.x_size_physical(), padded_grid.y_size_physical());
        std::copy(result.begin(), result.end(), result_grid.data());
        
        // Remove margins
        auto unpadded_result = result_grid.template with_removed_margins<margin, margin>();
        
        // Calculate checksum from result
        std::string checksum = unpadded_result.get_checksum();
        
        return { execution_time.count(), checksum };
    }
};

} // namespace cellato::run

#endif // REFERENCE_IMPL_MANAGER_HPP
