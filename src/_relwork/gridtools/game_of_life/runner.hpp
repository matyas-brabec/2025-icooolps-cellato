#ifndef GRIDTOOLS_GAME_OF_LIFE_RUNNER_HPP
#define GRIDTOOLS_GAME_OF_LIFE_RUNNER_HPP

#include <cstddef>
#include <cstdint>

#include <memory>
#include <stdexcept>
#include <vector>

#include "experiments/run_params.hpp"

#ifndef ENABLE_GRIDTOOLS
#error "GridTools is not enabled, this source file should not be compiled."
#endif // ENABLE_GRIDTOOLS

namespace gridtools::game_of_life {

struct real_runner {
    using value_type = std::uint8_t;

    virtual ~real_runner() = default;

    virtual void init(int* grid,
                  const cellato::run::run_params& params) = 0;
    virtual void run(int steps) = 0;
    virtual std::vector<int> fetch_result() = 0;
};

std::unique_ptr<real_runner> create_CUDA_runner();
std::unique_ptr<real_runner> create_CPU_runner();

struct runner {
    using value_type = std::uint8_t;

    void init(int* grid,
              const cellato::run::run_params& params) {
        if (!real_runner_) {
            if (params.device == "CPU") {
                real_runner_ = create_CPU_runner();
            } else if (params.device == "CUDA") {
                real_runner_ = create_CUDA_runner();
            } else {
                throw std::runtime_error("Unsupported device: " + params.device);
            }
        }

        if (!real_runner_) {
            throw std::runtime_error("Failed to create real_runner instance");
        }

        real_runner_->init(grid, params);
    }

    void run(int steps) {
        real_runner_->run(steps);
    }

    std::vector<int> fetch_result() {
        return real_runner_->fetch_result();
    }

private:
    std::unique_ptr<real_runner> real_runner_;
};

} // namespace gridtools::game_of_life

#endif // GRIDTOOLS_GAME_OF_LIFE_RUNNER_HPP
