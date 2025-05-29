#ifndef EXPERIMENT_MANAGER_HPP
#define EXPERIMENT_MANAGER_HPP

#include <vector>
#include <thread>
#include <iostream>

#include "./run_params.hpp"
#include "./experiment_report.hpp"

namespace cellato::run {

#define LOG std::cerr

using namespace cellato::memory::grids;

template <typename cell_t>
using print_config = cellato::memory::grids::standard::print_config<cell_t>;

template <typename test_suite>
class experiment_manager {

public:
    using original_cell_t = typename test_suite::original_cell_t;
    using grid_store_word_t = typename test_suite::grid_store_word_t;

    using grid_t = typename test_suite::grid_t;
    using traverser_t = typename test_suite::traverser_t;
    using run_params = cellato::run::run_params;

    using standard_grid_t = cellato::memory::grids::standard::grid<original_cell_t>;

    experiment_manager() = default;

    experiment_report run_experiment(const run_params& params, const std::vector<original_cell_t>& initial_state) {
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

    void set_print_config(print_config<original_cell_t> config) {
        _print_config = config;
    }

    private:
    print_config<original_cell_t> _print_config;

    std::tuple<double, std::string> run_round(int round, const run_params& params, 
                                              const std::vector<original_cell_t>& initial_state) {

        if (round < params.warmup_rounds) {
            LOG << "\nWarmup round: " << round << "\n";
        }
        else {
            LOG << "\nRound: " << round - params.warmup_rounds << "\n";
        }
        
        grid_t grid = get_padded_grid(params, initial_state);
        traverser_t traverser = get_initialized_traverser(grid, params);

        auto execution_time = run_traverser(traverser, params);

        grid_t result = traverser.fetch_result();
        auto result_as_standard = result
            .to_standard()
            .template with_removed_margins<test_suite::x_margin, test_suite::y_margin>();

        return { execution_time, result_as_standard.get_checksum() };
    }

    grid_t get_padded_grid(const run_params& params, const std::vector<original_cell_t>& initial_state) {
        standard_grid_t initial_grid(params.x_size, params.y_size);
        std::copy(initial_state.begin(), initial_state.end(), initial_grid.data());

        auto grid_padded = initial_grid.template with_empty_margins<test_suite::x_margin, test_suite::y_margin>();

        grid_t grid{grid_padded};
        return grid;
    }

    double run_traverser(traverser_t& traverser, const run_params& params) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (params.print) {
            traverser.run(params.steps, 
                [&](int iter, const auto& grid) {
                    auto standard_grid = grid
                    .to_standard()
                    .template with_removed_margins<test_suite::x_margin, test_suite::y_margin>();

                    LOG << "\nIteration: " << iter << "\n";
                    standard_grid.print(LOG, _print_config);

                    std::this_thread::sleep_for(std::chrono::milliseconds(400));
                    LOG << "\n";
                }
            );
        }
        else {
            traverser.run(params.steps);
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> execution_time = end_time - start_time;
        return execution_time.count();
    }

    traverser_t get_initialized_traverser(grid_t& grid, 
                                          const cellato::run::run_params& params) {
        traverser_t traverser;
        traverser.init(grid, params);
        return traverser;
    }
};

}

#endif // EXPERIMENT_MANAGER_HPP