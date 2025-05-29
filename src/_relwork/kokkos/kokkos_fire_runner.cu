#include "fire/runner.hpp"

#include "Kokkos_Core.hpp"

namespace kokkos::fire {


struct fire_runner : public real_runner {
public:
    using value_type = std::uint8_t;

    void init(int* grid,
              const cellato::run::run_params& params) {
    
        std::size_t x_size = params.x_size;
        std::size_t y_size = params.y_size;

        grid_ = Kokkos::View<value_type**>("grid", x_size, y_size);
        next_grid_ = Kokkos::View<value_type**>("next_grid", x_size, y_size);
        // Initialize the Kokkos view with the provided grid data
        for (std::size_t i = 0; i < x_size; ++i) {
            for (std::size_t j = 0; j < y_size; ++j) {
                grid_(i, j) = grid[i * y_size + j];
            }
        }
    }

    void run(int steps) {
        for (int step = 0; step < steps; ++step) {
            run_step();
        }
    }

    std::vector<int> fetch_result() {
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

    void run_step() {
        Kokkos::parallel_for("FireStep", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {grid_.extent(0) - 1, grid_.extent(1) - 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            // fire rules
            next_grid_(i, j) = /* apply rules */ grid_(i, j); // Placeholder for actual fire rules logic
        });

        using std::swap;
        swap(grid_, next_grid_);
    }

private:
    Kokkos::View<value_type**> grid_;
    Kokkos::View<value_type**> next_grid_;
};

std::unique_ptr<real_runner> create_runner() {
    return std::make_unique<fire_runner>();
}

} // namespace kokkos::fire
