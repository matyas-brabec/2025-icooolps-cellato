#ifndef RELWORK_RUNNER_WRAPPER_HPP
#define RELWORK_RUNNER_WRAPPER_HPP

#include "experiments/run_params.hpp"
#include <vector>
#include <cstddef>

namespace relwork {

template <typename automaton_config, typename relwork_framework_runner>
struct runner_wrapper {

    using cell_state = typename automaton_config::cell_state;

    void init(const cell_state* grid,
              const cellib::run::run_params& params = cellib::run::run_params()) {

        auto x_size = static_cast<std::size_t>(params.x_size);
        auto y_size = static_cast<std::size_t>(params.y_size);

        _current_grid.resize(x_size * y_size);

        for (std::size_t i = 0; i < x_size * y_size; ++i) {
            _current_grid[i] = static_cast<int>(grid[i]);
        }

        _runner.init(_current_grid.data(), params);
    }

    void init_cuda() {
        // nothing to do here
    }

    void run(int steps) {
        _runner.run(steps);
    }

    void run_on_cuda(int steps) {
        _runner.run(steps);
    }

    std::vector<cell_state> fetch_result() {
        auto result = _runner.fetch_result();
        std::vector<cell_state> result_converted(result.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result_converted[i] = static_cast<cell_state>(result[i]);
        }

        return result_converted;
    }

    private:
    
    relwork_framework_runner _runner;
    std::vector<int> _current_grid;
};

} // namespace relwork

#endif // RELWORK_RUNNER_WRAPPER_HPP