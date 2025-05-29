#ifndef __FRAMEWORK_OF_CHOICE__FIRE_RUNNER_HPP
#define __FRAMEWORK_OF_CHOICE__FIRE_RUNNER_HPP

#include <cstddef>
#include "experiments/run_params.hpp"

#include <vector>

namespace __framework_of_choice__::fire {

struct runner {
    void init(int* grid,
              const cellato::run::run_params& params) {
        // ...
        (void)grid;
        (void)params;
    }

    void run(int steps) {
        // ...
        (void)steps;
    }

    std::vector<int> fetch_result() {
        // ...
        return {};
    }
};

}

#endif // __FRAMEWORK_OF_CHOICE__FIRE_RUNNER_HPP