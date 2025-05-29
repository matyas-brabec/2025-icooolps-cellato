#ifndef KOKKOS_WIRE_RUNNER_HPP
#define KOKKOS_WIRE_RUNNER_HPP

#include <cstddef>
#include <cstdint>

#include <vector>
#include "experiments/run_params.hpp"

#ifndef ENABLE_KOKKOS
#error "Kokkos is not enabled, this source file should not be compiled."
#endif // ENABLE_KOKKOS

namespace kokkos::wire {

struct runner {
    using value_type = std::uint8_t;

    void init(int* grid,
              const cellato::run::run_params& params) {
        // ...
    }

    void run(int steps) {
        // ...
    }

    std::vector<int> fetch_result() {
        // ...

        return {};
    }
};

} // namespace kokkos::wire

#endif // KOKKOS_WIRE_RUNNER_HPP
