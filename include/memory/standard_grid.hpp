#ifndef CELLIB_MEMORY_STANDARD_GRID_HPP
#define CELLIB_MEMORY_STANDARD_GRID_HPP

#include <cstddef>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cassert>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdint>

#include "./interface.hpp"

namespace cellib::memory::grids::standard {
template <typename cell_type>
class print_config;

template <typename cell_type>
struct cuda_params {
    cell_type* cuda_data;
    std::size_t x_size;
    std::size_t y_size;
};

template <typename cell_type, device device_type = device::CPU>
class grid {
public:
    using store_type = cell_type;
    constexpr static bool HAS_OWN_PRINT = true;

    grid(std::size_t x_size, std::size_t y_size)
        : _properties{x_size, y_size}, _data(x_size * y_size) {}

    grid() = default;

    grid(std::vector<cell_type>&& data, std::size_t x_size, std::size_t y_size)
        : _properties{x_size, y_size}, _data(std::move(data)) {
        if (_data.size() != x_size * y_size) {
            throw std::invalid_argument("Data size does not match grid dimensions");
        }
    }

    grid(cuda_params<cell_type> params)
        : _properties{params.x_size, params.y_size}, _data() {

        static_assert(device_type == device::CUDA, "This constructor is only for CUDA device");

        _cuda_data = params.cuda_data;
    }

    grid(const grids::properties& properties, std::vector<cell_type> data)
        : _properties(properties), _data(std::move(data)) {}
    
    cell_type* data() const {
        if constexpr (device_type == device::CUDA) {
            return _cuda_data;
        } else {
            return const_cast<cell_type*>(_data.data());
        }
    }

    cell_type* data() {
        if constexpr (device_type == device::CUDA) {
            return _cuda_data;
        } else {
            return _data.data();
        }
    }

    std::size_t x_size_physical() const {
        return _properties.x_size;
    }

    std::size_t y_size_physical() const {
        return _properties.y_size;
    }

    std::size_t x_size_logical() const {
        return _properties.x_size;
    }

    std::size_t y_size_logical() const {
        return _properties.y_size;
    }

    template <int x_margin, int y_margin>
    grid<cell_type> with_empty_margins() const {
        static_assert(device_type == device::CPU, "This function is only for CPU device");

        grids::properties new_properties {
            _properties.x_size + 2 * x_margin,
            _properties.y_size + 2 * y_margin
        };

        std::vector<cell_type> new_data(new_properties.x_size * new_properties.y_size, cell_type{});

        for (std::size_t y = 0; y < _properties.y_size; ++y) {
            for (std::size_t x = 0; x < _properties.x_size; ++x) {
                new_data[new_properties.idx(x + x_margin, y + y_margin)] = _data[_properties.idx(x, y)];
            }
        }

        return grid<cell_type>(new_properties, std::move(new_data));
    }

    template <int x_margin, int y_margin>
    grid<cell_type> with_removed_margins() const {
        static_assert(device_type == device::CPU, "This function is only for CPU device");

        grids::properties new_properties {
            _properties.x_size - 2 * x_margin,
            _properties.y_size - 2 * y_margin
        };

        std::vector<cell_type> new_data(new_properties.x_size * new_properties.y_size);

        for (std::size_t y = 0; y < new_properties.y_size; ++y) {
            for (std::size_t x = 0; x < new_properties.x_size; ++x) {
                new_data[new_properties.idx(x, y)] = _data[_properties.idx(x + x_margin, y + y_margin)];
            }
        }

        return grid<cell_type>(new_properties, std::move(new_data));
    }

    grid<cell_type> to_standard() const {
        return *this;
    }

    void print(std::ostream& os, print_config<cell_type> config = print_config<cell_type>()) const {
        static_assert(device_type == device::CPU, "This function is only for CPU device");

        for (std::size_t y = 0; y < _properties.y_size; ++y) {
            for (std::size_t x = 0; x < _properties.x_size; ++x) {
                os << config.get_str(_data[_properties.idx(x, y)]) << " ";
            }
            os << "\n";
        }
    }

    grid<cell_type, device::CPU> to_cpu() const {
        static_assert(device_type == device::CUDA, "This function is only for CUDA device");

        std::vector<cell_type> host_data(_properties.x_size * _properties.y_size);
        cudaMemcpy(host_data.data(), _cuda_data, host_data.size() * sizeof(cell_type), cudaMemcpyDeviceToHost);

        return grid<cell_type, device::CPU>{_properties, std::move(host_data)};

    }

    grid<cell_type, device::CUDA> to_cuda() const {
        static_assert(device_type == device::CPU, "This function is only for CPU device");

        cell_type* device_data;
        cudaMalloc((void**)&device_data, _data.size() * sizeof(cell_type));
        cudaMemcpy(device_data, _data.data(), _data.size() * sizeof(cell_type), cudaMemcpyHostToDevice);

        cuda_params<cell_type> params{
            device_data,
            _properties.x_size,
            _properties.y_size
        };

        return grid<cell_type, device::CUDA>(params);
    }

    void free_cuda_memory() {
        if constexpr (device_type == device::CUDA) {
            if (_cuda_data) {
                cudaFree(_cuda_data);
                _cuda_data = nullptr;
            }
        }
    }

    std::string get_checksum() const {
        static_assert(device_type == device::CPU, "This function is only for CPU device");

        constexpr int dims = 4;
        std::stringstream result;
        
        // Calculate tile dimensions
        const std::size_t tile_width = (_properties.x_size + dims - 1) / dims;
        const std::size_t tile_height = (_properties.y_size + dims - 1) / dims;
        
        // Process each tile
        for (int tile_y = 0; tile_y < dims; ++tile_y) {
            for (int tile_x = 0; tile_x < dims; ++tile_x) {
                // Calculate the boundaries of this tile
                std::size_t start_x = tile_x * tile_width;
                std::size_t start_y = tile_y * tile_height;
                std::size_t end_x = std::min(start_x + tile_width, _properties.x_size);
                std::size_t end_y = std::min(start_y + tile_height, _properties.y_size);
                
                // Sum the values in this tile
                std::uint64_t sum = 0;
                for (std::size_t y = start_y; y < end_y; ++y) {
                    for (std::size_t x = start_x; x < end_x; ++x) {
                        sum += static_cast<std::uint64_t>(_data[_properties.idx(x, y)]);
                    }
                }
                
                // Add to result string with hyphen separator (except for first value)
                if (tile_y > 0 || tile_x > 0) {
                    result << "-";
                }
                result << sum;
            }
        }
        
        return result.str();
    }

private:

    grids::properties _properties;
    std::vector<cell_type> _data;

    cell_type* _cuda_data;
};

template <typename cell_type>
class print_config {
public:
    print_config& with(cell_type state, const std::string& symbol) {
        _state_to_symbol[state] = symbol;
        return *this;
    }

    std::string get_str(cell_type state) const {
        auto it = _state_to_symbol.find(state);
        
        if (it != _state_to_symbol.end()) {
            return it->second;
        }
        
        auto state_as_int = static_cast<int>(state);
        return std::to_string(state_as_int);
    }

    static print_config<cell_type> empty() {
        return print_config<cell_type>();
    }

private:
    std::map<cell_type, std::string> _state_to_symbol;
};

}

#endif // CELLIB_MEMORY_STANDARD_GRID_HPP