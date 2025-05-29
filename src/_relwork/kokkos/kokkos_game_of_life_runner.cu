#include "game_of_life/runner.hpp"

#include "Kokkos_Core.hpp"


namespace kokkos::game_of_life {

struct game_of_life_runner : public real_runner {
    using value_type = std::uint8_t;

    static inline enum class space {
        CPU,
        CUDA
    } execution_space = space::CPU;

    void init(int* grid,
              const cellato::run::run_params& params) override {
        // Initialize Kokkos
        kokkos_initialize(params);

        std::size_t x_size = params.x_size;
        std::size_t y_size = params.y_size;

        grid_ = Kokkos::View<value_type**, Kokkos::SharedSpace>("grid", x_size, y_size);
        next_grid_ = Kokkos::View<value_type**, Kokkos::SharedSpace>("next_grid", x_size, y_size);

        // Initialize the Kokkos view with the provided grid data
        for (std::size_t i = 0; i < x_size; ++i) {
            for (std::size_t j = 0; j < y_size; ++j) {
                grid_(i, j) = grid[i * y_size + j];
            }
        }
    }

    void run(int steps) override {
        for (int step = 0; step < steps; ++step) {
            if (execution_space == space::CPU) {
                run_step_cpu();
            } else if (execution_space == space::CUDA) {
                run_step_cuda();
            } else {
                throw std::runtime_error("Unsupported execution space");
            }
        }
    }

    std::vector<int> fetch_result() override {
        // ...

        std::vector<int> result;
        result.reserve(grid_.extent(0) * grid_.extent(1));

        for (std::size_t i = 0; i < grid_.extent(0); ++i) {
            for (std::size_t j = 0; j < grid_.extent(1); ++j) {
                result.push_back(static_cast<int>(grid_(i, j)));
            }
        }

        return result;
    }

    void run_step_cpu() {
        Kokkos::parallel_for("GoLStepCPU", Kokkos::MDRangePolicy(Kokkos::Serial(), {1, 1}, {grid_.extent(0) - 1, grid_.extent(1) - 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            // game_of_life rules
            auto neighbors = grid_(i - 1, j - 1) + grid_(i - 1, j) + grid_(i - 1, j + 1) +
                             grid_(i, j - 1) + grid_(i, j + 1) +
                             grid_(i + 1, j - 1) + grid_(i + 1, j) + grid_(i + 1, j + 1);
            next_grid_(i, j) = grid_(i, j) ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
        });

        using std::swap;
        swap(grid_, next_grid_);
    }

    void run_step_cuda() {
        Kokkos::parallel_for("GoLStepCUDA", Kokkos::MDRangePolicy(Kokkos::Cuda(), {1, 1}, {grid_.extent(0) - 1, grid_.extent(1) - 1}, /* tiling */ {16, 16}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            // game_of_life rules
            auto neighbors = grid_(i - 1, j - 1) + grid_(i - 1, j) + grid_(i - 1, j + 1) +
                             grid_(i, j - 1) + grid_(i, j + 1) +
                             grid_(i + 1, j - 1) + grid_(i + 1, j) + grid_(i + 1, j + 1);
            next_grid_(i, j) = grid_(i, j) ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
        });

        Kokkos::fence(); // Ensure all operations are complete before swapping

        using std::swap;
        swap(grid_, next_grid_);
    }

private:
    void kokkos_initialize(const cellato::run::run_params& params) {
        static struct kokkos_init_guard {
            kokkos_init_guard(const cellato::run::run_params& params, space *space = nullptr) {
                Kokkos::InitializationSettings init_settings_;

                if (params.device == "CPU") {
                    if (space) {
                        *space = space::CPU;
                    }

                    init_settings_.set_num_threads(1);
                } else if (params.device == "CUDA") {
                    if (space) {
                        *space = space::CUDA;
                    }
                    init_settings_.set_device_id(0);
                } else {
                    throw std::runtime_error("Unsupported device: " + params.device);
                }

                Kokkos::initialize(init_settings_);
            }

            ~kokkos_init_guard() {
                Kokkos::finalize();
            }
        } guard(params, &execution_space);
    }

    Kokkos::View<value_type**, Kokkos::SharedSpace> grid_;
    Kokkos::View<value_type**, Kokkos::SharedSpace> next_grid_;
};

std::unique_ptr<real_runner> create_runner() {
    return std::make_unique<game_of_life_runner>();
}

} // namespace kokkos::game_of_life
