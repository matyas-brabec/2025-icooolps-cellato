#ifndef CELLATO_BIT_ARRAY_GRID_HPP
#define CELLATO_BIT_ARRAY_GRID_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <bitset>
#include <cuda_runtime.h>

#include "interface.hpp"
#include "standard_grid.hpp"
#include "grid_utils.hpp"
#include "state_dictionary.hpp"

namespace cellato::memory::grids::bit_array {

using namespace cellato::memory::grids::utils;
using namespace cellato::memory::grids;

// Define CUDA parameters for bit array grid
template <typename store_word_type>
struct cuda_params {
    store_word_type* cuda_data;
    std::size_t x_size;
    std::size_t y_size;
    std::size_t bits_per_cell;
};

// Proxy class that provides array-like access to bit_array elements
template <typename states_dict_t, typename store_word_type>
class BitArrayProxy {
private:
    store_word_type* _data;
    static constexpr int _bits_per_cell = states_dict_t::needed_bits;
    static constexpr store_word_type _cell_mask = (1 << _bits_per_cell) - 1;
    
public:

    // Reference proxy class to allow both read and write operations
    static constexpr int cells_per_word = sizeof(store_word_type) * 8 / _bits_per_cell;
    
    class CellReference {
    private:
        BitArrayProxy& _proxy;
        std::size_t _index;

    public:
        CUDA_CALLABLE CellReference(BitArrayProxy& proxy, std::size_t index)
            : _proxy(proxy), _index(index) {}

        // Implicit conversion operator for reading
        CUDA_CALLABLE operator store_word_type() const {
            // Calculate which word and which bits within that word
            std::size_t word_index = _index / _proxy.cells_per_word;
            std::size_t bit_offset = (_index % _proxy.cells_per_word) * _proxy._bits_per_cell;
            
            // Extract bits for this cell
            store_word_type state_index = (_proxy._data[word_index] >> bit_offset) & _proxy._cell_mask;
            
            // Convert back to enum state
            return state_index;
        }

    };

    CUDA_CALLABLE BitArrayProxy(store_word_type* data)
        : _data(data) {}

    // Operator[] returns a reference proxy for both read and write access
    CUDA_CALLABLE CellReference get_individual_cell_at(std::size_t index) {
        return CellReference(*this, index);
    }
    
    // Const version for read-only access
    CUDA_CALLABLE store_word_type& operator[](std::size_t index) {
        return _data[index];
    }
};

template <typename states_dictionary_t, typename store_word_type = std::uint32_t, device device_type = device::CPU>
class grid {
public:
    using states_dict_t = states_dictionary_t;

    constexpr static bool HAS_OWN_PRINT = false;

    // Number of bits needed to represent each cell state
    constexpr static int bits_per_cell = states_dict_t::needed_bits;
    
    // How many cells can be packed into a single word
    constexpr static int cells_per_word = sizeof(store_word_type) * 8 / bits_per_cell;
    
    // Bit mask for a single cell
    constexpr static store_word_type cell_mask = (1 << bits_per_cell) - 1;

    using original_state_t = typename states_dict_t::state_t;
    using cell_t = original_state_t;
    using store_type = store_word_type;
    using cuda_params_t = cuda_params<store_word_type>;

    using cell_ptr_t = BitArrayProxy<states_dict_t, store_word_type>;

    friend class cellato::memory::grids::bit_array::grid<states_dict_t, store_word_type, device::CPU>;
    friend class cellato::memory::grids::bit_array::grid<states_dict_t, store_word_type, device::CUDA>;

    // Default constructor
    grid() = default;

    // Constructor for creating empty grid of specified size
    grid(std::size_t y_size, std::size_t x_size)
        : _x_size(x_size), _y_size(y_size) {

        assert_valid_dimensions(x_size, y_size);

        if constexpr (device_type == device::CPU) {
            // Calculate number of words needed
            std::size_t total_cells = x_size * y_size;
            std::size_t words_needed = (total_cells + cells_per_word - 1) / cells_per_word;
            _data.resize(words_needed, 0);
        }
    }

    // Constructor for populating grid from existing data
    grid(std::size_t y_size, std::size_t x_size, const original_state_t* grid_input) 
        : _x_size(x_size), _y_size(y_size) {

        assert_valid_dimensions(x_size, y_size);

        static_assert(device_type == device::CPU, "This constructor is only for CPU device");
        
        std::size_t total_cells = x_size * y_size;
        std::size_t words_needed = (total_cells + cells_per_word - 1) / cells_per_word;
        _data.resize(words_needed, 0);
        
        // Populate the bit array
        for (std::size_t i = 0; i < total_cells; ++i) {
            set_cell_at_index(i, grid_input[i]);
        }
    }

    // Constructor from standard grid
    grid(const cellato::memory::grids::standard::grid<original_state_t>& standard_grid) {

        assert_valid_dimensions(standard_grid.x_size_physical(), standard_grid.y_size_physical());

        static_assert(device_type == device::CPU, "This constructor is only for CPU device");
        
        _x_size = standard_grid.x_size_physical();
        _y_size = standard_grid.y_size_physical();
        
        std::size_t total_cells = _x_size * _y_size;
        std::size_t words_needed = (total_cells + cells_per_word - 1) / cells_per_word;
        _data.resize(words_needed, 0);
        
        // Populate from standard grid
        const original_state_t* source_data = standard_grid.data();
        for (std::size_t i = 0; i < total_cells; ++i) {
            set_cell_at_index(i, source_data[i]);
        }
    }

    // CUDA-specific constructor
    grid(cuda_params_t params)
        : _x_size(params.x_size), _y_size(params.y_size) {
        
        assert_valid_dimensions(params.x_size, params.y_size);

        static_assert(device_type == device::CUDA, "This constructor is only for CUDA device");
        
        _cuda_data = params.cuda_data;
    }

    // Free CUDA memory
    void free_cuda_memory() {
        if constexpr (device_type == device::CUDA) {
            if (_cuda_data) {
                cudaFree(_cuda_data);
                _cuda_data = nullptr;
            }
        }
    }

    // Get cell state at specific coordinates
    original_state_t get_cell(std::size_t x, std::size_t y) const {
        static_assert(device_type == device::CPU, "get_cell is only supported on CPU");
        
        if (x >= x_size_logical() || y >= y_size_logical()) {
            throw std::out_of_range("Cell coordinates out of range");
        }
        
        std::size_t idx = y * _x_size + x;
        return get_cell_at_index(idx);
    }

    // Convert to vector of original states
    std::vector<original_state_t> to_original_representation() const {
        static_assert(device_type == device::CPU, "to_original_representation is only supported on CPU");
        
        std::vector<original_state_t> result(_x_size * _y_size);
        
        for (std::size_t i = 0; i < _x_size * _y_size; ++i) {
            result[i] = get_cell_at_index(i);
        }
        
        return result;
    }

    // Convert to standard grid
    cellato::memory::grids::standard::grid<original_state_t> to_standard() const {
        static_assert(device_type == device::CPU, "to_standard is only supported on CPU");
        
        auto grid_data = to_original_representation();
        return cellato::memory::grids::standard::grid<original_state_t>(
            std::move(grid_data), _x_size, _y_size);
    }

    // Move to CUDA device
    grid<states_dict_t, store_word_type, device::CUDA> to_cuda() const {
        static_assert(device_type == device::CPU, "to_cuda is only supported on CPU");
        
        store_word_type* device_data;
        std::size_t data_size = _data.size() * sizeof(store_word_type);
        
        cudaMalloc((void**)&device_data, data_size);
        cudaMemcpy(device_data, _data.data(), data_size, cudaMemcpyHostToDevice);
        
        cuda_params_t params{
            device_data,
            _x_size,
            _y_size,
            bits_per_cell
        };
        
        return grid<states_dict_t, store_word_type, device::CUDA>(params);
    }
    
    // Move from CUDA to CPU
    grid<states_dict_t, store_word_type, device::CPU> to_cpu() const {
        static_assert(device_type == device::CUDA, "to_cpu is only supported on CUDA");
        
        std::size_t total_cells = _x_size * _y_size;
        std::size_t words_needed = (total_cells + cells_per_word - 1) / cells_per_word;
        
        // Create a CPU grid
        grid<states_dict_t, store_word_type, device::CPU> cpu_grid(_y_size, _x_size);

        // Copy data from device to host
        cudaMemcpy(cpu_grid._data.data(), _cuda_data, words_needed * sizeof(store_word_type), cudaMemcpyDeviceToHost);
        
        return cpu_grid;
    }

    // Size methods
    std::size_t x_size_logical() const { return _x_size; }
    std::size_t y_size_logical() const { return _y_size; }
    std::size_t x_size_physical() const { return _x_size / cells_per_word; }
    std::size_t y_size_physical() const { return _y_size; }

    // Access to underlying data
    auto data() {
        if constexpr (device_type == device::CUDA) {
            return BitArrayProxy<states_dict_t, store_word_type>(_cuda_data);
        } else {
            return BitArrayProxy<states_dict_t, store_word_type>(_data.data());
        }
    }
    
    auto data() const {
        if constexpr (device_type == device::CUDA) {
            return BitArrayProxy<states_dict_t, store_word_type>(_cuda_data);
        } else {
            return BitArrayProxy<states_dict_t, store_word_type>(const_cast<store_word_type*>(_data.data()));
        }
    }

    // Keep the old methods with a different name for when we need the raw pointer
    store_word_type* raw_data() {
        if constexpr (device_type == device::CUDA) {
            return _cuda_data;
        } else {
            return _data.data();
        }
    }
    
    store_word_type* raw_data() const {
        if constexpr (device_type == device::CUDA) {
            return _cuda_data;
        } else {
            return const_cast<store_word_type*>(_data.data());
        }
    }

private:
    std::vector<store_word_type> _data;   // Used for CPU storage
    store_word_type* _cuda_data = nullptr; // Used for CUDA storage
    std::size_t _x_size, _y_size;        // Grid dimensions

    // Helper method to set cell value at a specific index
    void set_cell_at_index(std::size_t index, original_state_t state) {
        std::size_t word_index = index / cells_per_word;
        std::size_t bit_offset = (index % cells_per_word) * bits_per_cell;
        
        // Clear existing bits at position
        _data[word_index] &= ~(cell_mask << bit_offset);
        
        // Set new bits
        std::size_t state_index = states_dict_t::state_to_index(state);
        _data[word_index] |= (state_index & cell_mask) << bit_offset;
    }
    
    // Helper method to get cell value at a specific index
    original_state_t get_cell_at_index(std::size_t index) const {
        std::size_t word_index = index / cells_per_word;
        std::size_t bit_offset = (index % cells_per_word) * bits_per_cell;
        
        // Extract bits for this cell
        std::size_t state_index = (_data[word_index] >> bit_offset) & cell_mask;
        
        return states_dict_t::index_to_state(state_index);
    }

    void assert_valid_dimensions(std::size_t x_size, std::size_t y_size) const {
        (void) y_size;

        if (x_size % cells_per_word != 0) {
            throw std::invalid_argument("Grid x_size must be divisible by cells_per_word. x_size=" + 
                                       std::to_string(x_size) + ", cells_per_word=" + 
                                       std::to_string(cells_per_word) +
                                       " (bits [" + std::to_string(sizeof(store_word_type) * 8) + 
                                       "] / bits_per_cell [" + std::to_string(bits_per_cell) + "])");
        } 
    }
};

} // namespace cellato::memory::grids::bit_array

#endif // CELLATO_BIT_ARRAY_GRID_HPP