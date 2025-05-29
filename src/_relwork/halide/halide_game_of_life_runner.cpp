#include "game_of_life/runner.hpp"

#include "Halide.h"

using namespace Halide;

namespace halide::game_of_life {

struct game_of_life_runner : public real_runner {
    using value_type = std::uint8_t;

    void init(int* in_grid,
              const cellato::run::run_params& params) override {
        try {
            current_grid = Buffer<value_type>(params.x_size, params.y_size);
            next_grid = Buffer<value_type>(params.x_size, params.y_size);

            neighbors(x, y) = grid(x - 1, y - 1) + grid(x - 1, y) + grid(x - 1, y + 1) +
                grid(x, y - 1) + grid(x, y + 1) +
                grid(x + 1, y - 1) + grid(x + 1, y) + grid(x + 1, y + 1);

            step(x, y) = select(grid(x, y) == cast<value_type>(1), // Alive cell
                select(neighbors(x, y) == 2 || neighbors(x, y) == 3, cast<value_type>(1), cast<value_type>(0)), // Alive cell rules
                select(neighbors(x, y) == 3, cast<value_type>(1), cast<value_type>(0)) // Dead cell rules
            );

            clamp = BoundaryConditions::constant_exterior(
                step, cast<value_type>(0), {
                    {1, params.x_size - 2},
                    {1, params.y_size - 2}
                }
            );

            // Initialize the grid with the input data
            for (int i = 0; i < params.x_size; ++i) {
                for (int j = 0; j < params.y_size; ++j) {
                    current_grid(i, j) = in_grid[i * params.y_size + j];
                }
            }

            clamp.set_estimates({
                {0, params.x_size},
                {0, params.y_size}
            });

            if (params.device == "CPU") {
                // Set the target to CPU
                Target target = get_host_target();
                clamp.compile_jit(target);
            } else if (params.device == "CUDA") {
                // Set the target to CUDA
                Target target = get_host_target();
                target.set_feature(Target::CUDA);
                if (!target.has_gpu_feature()) {
                    throw std::runtime_error("CUDA feature is not available on this target.");
                }

                clamp.gpu_tile(x, y, xi, yi, xo, yo, 16, 16);
                clamp.compile_jit(target);
            } else {
                throw std::runtime_error("Unsupported device: " + params.device);
            }
        } catch (const Halide::CompileError& e) {
            throw std::runtime_error("Failed to compile Halide function for GPU: " + std::string(e.what()));
        }
    }

    void run(int steps) override {
        try {
            for (int s = 0; s < steps; ++s) {
                    grid.set(current_grid);
                    clamp.realize(next_grid);
                // Swap the grids
                std::swap(current_grid, next_grid);
            }
        } catch (const Halide::RuntimeError& e) {
            throw std::runtime_error("Halide runtime error: " + std::string(e.what()));
        } catch (const Halide::CompileError& e) {
            throw std::runtime_error("Halide compile error: " + std::string(e.what()));
        }
    }

    std::vector<int> fetch_result() override {
        std::vector<int> result;
        result.reserve(current_grid.width() * current_grid.height());

        current_grid.copy_to_host();

        for (int i = 0; i < current_grid.width(); ++i) {
            for (int j = 0; j < current_grid.height(); ++j) {
                result.push_back(current_grid(i, j));
            }
        }

        return result;
    }

private:
    Var x{"x"}, y{"y"};
    Var xi{"xi"}, yi{"yi"};
    Var xo{"xo"}, yo{"yo"};
    ImageParam grid{UInt(8), 2, "grid"};
    Func step{"step"}, neighbors{"neighbors"}, clamp{"clamp"};
    Buffer<value_type> current_grid, next_grid;
};

std::unique_ptr<real_runner> create_runner() {
    return std::make_unique<game_of_life_runner>();
}

} // namespace halide::game_of_life
