#ifndef CELLATO_TRAVERSERS_UTILS_HPP
#define CELLATO_TRAVERSERS_UTILS_HPP

#include <type_traits>


#ifndef CUDA_CALLABLE
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#endif


namespace cellato::traversers::utils {

template <typename T, typename = void>
struct has_save_to_method : std::false_type {};

template <typename T>
struct has_save_to_method<T, 
    std::void_t<decltype(std::declval<T>().save_to(
        std::declval<void*>(), 
        std::declval<std::size_t>()))>> 
    : std::true_type {};

template <typename grid_data_t, typename value_t>
CUDA_CALLABLE void save_to(grid_data_t grid, std::size_t index, value_t new_value) {

    if constexpr (has_save_to_method<value_t>::value) {
        new_value.save_to(grid, index);
    } else {
        grid[index] = new_value;
    }
}

}

#endif // CELLATO_TRAVERSERS_UTILS_HPP