#ifndef CELLATO_MEMORY_INTERFACE_HPP
#define CELLATO_MEMORY_INTERFACE_HPP

#ifndef CUDA_CALLABLE
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#endif

namespace cellato::memory::grids {

enum class device {
    CPU,
    CUDA
};

struct properties {
    std::size_t x_size;
    std::size_t y_size;

    CUDA_CALLABLE std::size_t idx(std::size_t x, std::size_t y) const {
        return y * x_size + x;
    }
};

struct point {
    std::size_t x;
    std::size_t y;
};

template <typename grid_data_type>
struct point_in_grid {

    point_in_grid() = default;
    CUDA_CALLABLE point_in_grid(grid_data_type grid_data)
        : grid(grid_data) {} 

    grid_data_type grid;

    grids::properties properties;

    grids::point position;

    CUDA_CALLABLE std::size_t idx() const {
        return properties.idx(position.x, position.y);
    }
};

}

#endif // CELLATO_MEMORY_INTERFACE_HPP