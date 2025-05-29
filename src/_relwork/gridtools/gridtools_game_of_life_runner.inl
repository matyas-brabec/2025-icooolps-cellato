#include "game_of_life/runner.hpp"

#include "gridtools/stencil/frontend/make_grid.hpp"
#include "gridtools/stencil/frontend/run.hpp"

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
namespace {
using stencil_backend_t = gridtools::stencil::gpu<>;
using storage_traits_t = gridtools::storage::gpu;
} // namespace
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
namespace {
using stencil_backend_t = gridtools::stencil::cpu_ifirst<>;
using storage_traits_t = gridtools::storage::cpu_ifirst;
} // namespace
#endif


#ifndef ENABLE_GRIDTOOLS
#error "GridTools is not enabled, this source file should not be compiled."
#endif // ENABLE_GRIDTOOLS

namespace {

namespace gt = gridtools;
namespace st = gt::stencil;

using axis_t = gt::stencil::axis<1>;
using exec_axis_t = gt::stencil::axis<1, gt::stencil::axis_config::offset_limit<0>>;

// Game of Life functor
struct life_functor {
    // Define accessors: one input (read-only) and one output (read-write)
    // The extent<-1,1,-1,1> indicates this functor will access neighbors 
    // 1 cell away in both i and j directions (8-neighbor stencil):contentReference[oaicite:2]{index=2}.
    using in  = st::cartesian::in_accessor<0, st::extent<-1, 1, -1, 1>>; 
    using out = st::cartesian::inout_accessor<1>;
    using param_list = st::make_param_list<in, out>;

    template <typename Eval>
    GT_FUNCTION static void apply(Eval && eval) {
        int center = eval(in());  // current cell state (0 or 1)
        // Sum all 8 neighbors around (i,j). We explicitly list offsets for Moore neighborhood.
        int neighbor_sum = eval(in(-1,-1)) + eval(in(-1,0)) + eval(in(-1, 1))
                         + eval(in( 0,-1))                  + eval(in( 0, 1))
                         + eval(in( 1,-1)) + eval(in( 1,0)) + eval(in( 1, 1));

        // Apply Game of Life rules:
        if (center == 1) {
            // Alive cell: survives if 2 or 3 neighbors, else dies
            eval(out()) = (neighbor_sum == 2 || neighbor_sum == 3) ? 1 : 0;
        } else {
            // Dead cell: becomes alive if exactly 3 neighbors
            eval(out()) = (neighbor_sum == 3) ? 1 : 0;
        }
    }
};

template <typename T>
auto make_storage(std::uintptr_t Nx, std::uintptr_t Ny, int halo) {
    // Create a storage builder with given dimensions and halo size
    return gt::storage::builder<storage_traits_t>
        .dimensions(Nx, Ny, 1)  // 1D for k dimension (not used here)
        .halos(halo, halo, 0)   // Halo for i,j (needed for neighbor access)
        .template type<T>()
        .build();
}

struct game_of_life_runner : public gridtools::game_of_life::real_runner {
private:
    using value_type = std::uint8_t;
    using storage_t = decltype(make_storage<value_type>(0, 0, 0));
    storage_t current_state;  // Current state of the grid
    storage_t next_state;     // Next state of the grid
    std::uintptr_t Nx = 0;            // Grid size in x dimension
    std::uintptr_t Ny = 0;            // Grid size in y dimension
    static constexpr int halo = 1;  // one-cell border (dead boundary)

public:
    void init(int* grid, const cellato::run::run_params& params) override {
        Nx = params.x_size;
        Ny = params.y_size;
        // Build storages with given dimensions and halo size
        current_state = make_storage<value_type>(Nx, Ny, halo);
        next_state = make_storage<value_type>(Nx, Ny, halo);

        // current_state = builder.name("current").initializer(
        //     [&](int i, int j, int k) { return input_array[j * Nx + i]; }
        // ).build();

        auto host_view = current_state->host_view();
        for (std::uintptr_t j = 0; j < Ny; ++j) {
            for (std::uintptr_t i = 0; i < Nx; ++i) {
                host_view(i, j, 0) = grid[j * Nx + i];  // Copy input grid data
            }
        }

    }

    void run(int steps) override {
        // Define iteration domain excluding the halo/border:
        gt::halo_descriptor halo_i(halo, halo, halo, Nx - halo - 1, Nx);
        gt::halo_descriptor halo_j(halo, halo, halo, Ny - halo - 1, Ny);
        auto grid_obj = st::make_grid(halo_i, halo_j, /*k_size=*/1);    

        for (int step = 0; step < steps; ++step) {
            // Create a stencil execution object with the current state and next state storages

            // Apply the Game of Life functor to the grid
            gt::stencil::run_single_stage(life_functor(), stencil_backend_t{}, 
                grid_obj, 
                current_state, next_state);

            // Swap current and next states for the next iteration
            std::swap(current_state, next_state);
        }
    }

    std::vector<int> fetch_result() override {
        std::vector<int> result(Nx * Ny, 0);  // Prepare output vector with zeros
        
        // Get a host view of the final state and copy it to result
        auto view = current_state->host_view();  // syncs device data to host if needed:contentReference[oaicite:10]{index=10}
        for (std::uintptr_t j = 0; j < Ny; ++j) {
            for (std::uintptr_t i = 0; i < Nx; ++i) {
                result[j * Nx + i] = view(i, j, 0);
            }
        }

        return result;
    }
};

} // namespace
