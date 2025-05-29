#ifndef CELLATO_CORE_VECTOR_INT_HPP
#define CELLATO_CORE_VECTOR_INT_HPP

#include <tuple>
#include <string>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "../memory/grid_utils.hpp"

// Add CUDA_CALLABLE macro definition
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace cellato::core::bitwise {

using namespace cellato::memory::grids::utils;

template <typename const_t>
struct constats_ops {

    static constexpr const_t ones = ~static_cast<const_t>(0);

    template <const_t value>
    CUDA_CALLABLE static constexpr int get_highest_set_bit() {
        if (value == 0) {
            return -1; // No bits are set
        }

        int highest_bit = 0;
        for (int i = 0; i < (int)sizeof(const_t) * 8; ++i) {
            if ((value >> i) & 1) {
                highest_bit = i;
            }
        }

        return highest_bit;
    }

    template <const_t value>
    CUDA_CALLABLE static constexpr const_t get_bit_row_at(int bit_idx) {
        return ((value >> bit_idx) & 1) * ones;
    }

    template <const_t value>
    CUDA_CALLABLE static constexpr const_t is_set_at(int bit_idx) {
        return ((value >> bit_idx) & 1) * ones;
    }
};

enum class bit_action {
    NO_ACTION = 0,
    SET_ZERO = 1,
    SET_ONE = 2,
    FLIP = 3,
};

struct op_shift_right {
    template <typename T, typename U>
    CUDA_CALLABLE static auto apply(T a, U b) {
        return a >> b;
    }

    template <int b, typename T>
    CUDA_CALLABLE static auto apply(T a) {
        return a >> b;
    }
};

struct op_shift_left {
    template <typename T, typename U>
    CUDA_CALLABLE static auto apply(T a, U b) {
        return a << b;
    }

    template <int b, typename T>
    CUDA_CALLABLE static auto apply(T a) {
        return a << b;
    }
};

struct op_and {
    template <typename T, typename U>
    CUDA_CALLABLE static auto apply(T a, U b) {
        return a & b;
    }

    template <typename const_t, const_t value>
    CUDA_CALLABLE static constexpr bit_action action_for_bit(int bit_idx) {
        auto bit_value = (value >> bit_idx) & 1;

        if (bit_value == 0) {
            return bit_action::SET_ZERO;
        } else {
            return bit_action::NO_ACTION;
        }
    }
};

struct op_or {
    template <typename T, typename U>
    CUDA_CALLABLE static auto apply(T a, U b) {
        return a | b;
    }

    template <typename const_t, const_t value>
    CUDA_CALLABLE static constexpr bit_action action_for_bit(int bit_idx) {
        auto bit_value = (value >> bit_idx) & 1;

        if (bit_value == 1) {
            return bit_action::SET_ONE;
        } else {
            return bit_action::NO_ACTION;
        }
    }
};

struct op_xor {
    template <typename T, typename U>
    CUDA_CALLABLE static auto apply(T a, U b) {
        return a ^ b;
    }

    template <typename const_t, const_t value>
    CUDA_CALLABLE static constexpr bit_action action_for_bit(int bit_idx) {
        auto bit_value = (value >> bit_idx) & 1;

        if (bit_value == 1) {
            return bit_action::FLIP;
        } else {
            return bit_action::NO_ACTION;
        }
    }
};

struct op_not {
    template <typename T>
    CUDA_CALLABLE static auto apply(T a) {
        return ~a;
    }
};

struct vector_int_factory;

template <typename vector_store_type, int bits>
class vector_int {
  public:
    template <typename T, int N>
    friend class vector_int;

    friend struct vector_int_factory;

    using store_t = repeated_tuple_t<vector_store_type, bits>;
    static constexpr int width_in_bits = sizeof(vector_store_type) * 8;

    template <int other_bits>
    using vector_int_higher_precision_t = vector_int<vector_store_type, (bits > other_bits ? bits : other_bits)>;

    static std::string type_info() {
        return "vector_int<" + std::to_string(bits) + ">";
    }

    template <typename val_t>
    CUDA_CALLABLE void set_at(int index, val_t value) {
        if (index > width_in_bits) {
            throw std::out_of_range("Index out of range");
        }

        for_each_bit([&]<std::size_t bit_idx>() {
            auto ith_bit = static_cast<vector_store_type>((value >> bit_idx) & 1);

            auto old_value = std::get<bit_idx>(numbers);
            auto new_value = (old_value & ~(1 << index)) | (ith_bit << index);

            std::get<bit_idx>(numbers) = new_value;
        });
    }

    template <typename val_t = int>
    CUDA_CALLABLE val_t get_at(int index) const {
        if (index > width_in_bits) {
            throw std::out_of_range("Index out of range");
        }

        val_t value = 0;

        for_each_bit([&]<std::size_t bit_idx>() {
            auto word = std::get<bit_idx>(numbers);
            auto bit = (word >> index) & 1;

            value |= (bit << bit_idx);
        });

        return value;
    }

    std::string to_str() const {
        std::string str;

        for (std::size_t i = 0; i < width_in_bits; ++i) {
            str += std::to_string(get_at(i)) + " ";
        }

        return str;
    }

    template <int other_bits>
    CUDA_CALLABLE vector_int_higher_precision_t<other_bits> get_added(
        vector_int<vector_store_type, other_bits> other) const {

        constexpr int res_bits = (bits > other_bits ? bits : other_bits);
        constexpr int min_bits = (bits < other_bits ? bits : other_bits);

        vector_int<vector_store_type, res_bits> result;

        std::get<0>(result.numbers) = std::get<0>(numbers) ^ std::get<0>(other.numbers);

        vector_store_type carry = std::get<0>(numbers) & std::get<0>(other.numbers);

        for_each_in<min_bits - 1>([&]<std::size_t i>() {
            constexpr auto next_bit_idx = i + 1;

            // Get bits from both vectors at the current position
            auto a = std::get<next_bit_idx>(numbers);
            auto b = std::get<next_bit_idx>(other.numbers);

            // XOR the bits and XOR with carry for the result
            auto bit_xor = a ^ b;
            std::get<next_bit_idx>(result.numbers) = bit_xor ^ carry;

            // Calculate new carry: (a & b) | (carry & (a ^ b))
            carry = (a & b) | (carry & bit_xor);
        });

        if constexpr (other_bits > bits) {
            // Fix: Properly propagate carry through all bits of the larger vector
            for_each_in<other_bits - bits>([&]<std::size_t i>() {
                constexpr auto bit_idx = min_bits + i;

                // Get bit from the larger vector
                auto a = std::get<bit_idx>(other.numbers);

                // XOR with carry for the result
                std::get<bit_idx>(result.numbers) = a ^ carry;

                // Update carry - if both the bit and current carry are 1, we need a new carry
                carry = carry & a;
            });
        } else if constexpr (other_bits < bits) {
            // Fix: Same correction for when the first vector is larger
            for_each_in<bits - other_bits>([&]<std::size_t i>() {
                constexpr auto bit_idx = min_bits + i;

                // Get bit from the larger vector
                auto a = std::get<bit_idx>(numbers);

                // XOR with carry for the result
                std::get<bit_idx>(result.numbers) = a ^ carry;

                // Update carry
                carry = carry & a;
            });
        }

        return result;
    }

    template <int other_bits>
    CUDA_CALLABLE vector_int_higher_precision_t<other_bits> get_ored(
        vector_int<vector_store_type, other_bits> other) const {

        return get_oped<other_bits, op_or>(other);
    }

    template <int other_bits>
    CUDA_CALLABLE vector_int_higher_precision_t<other_bits> get_xored(
        vector_int<vector_store_type, other_bits> other) const {

        return get_oped<other_bits, op_xor>(other);
    }

    template <int other_bits>
    CUDA_CALLABLE vector_int_higher_precision_t<other_bits> get_anded(
        vector_int<vector_store_type, other_bits> other) const {

        return get_oped<other_bits, op_and>(other);
    }

    CUDA_CALLABLE vector_int<vector_store_type, bits> mask_out_columns(
        vector_store_type mask) const {
        vector_int<vector_store_type, bits> result;

        for_each_in<bits>([&]<std::size_t i>() {
            auto word = std::get<i>(numbers);
            auto masked_word = word & mask;

            std::get<i>(result.numbers) = masked_word;
        });

        return result;
    }

    template <int target_bits>
    CUDA_CALLABLE vector_int<vector_store_type, target_bits> to_vector_with_bits() const {
        vector_int<vector_store_type, target_bits> result;
        constexpr auto min_bits = (target_bits < bits ? target_bits : bits);

        for_each_in<min_bits>([&]<std::size_t i>() {
            auto word = std::get<i>(numbers);
            std::get<i>(result.numbers) = word;
        });

        return result;
    }

    CUDA_CALLABLE vector_int<vector_store_type, bits> get_right_shifted_vector(int shift) const {
        return get_shifted_vector<op_shift_right>(shift);
    }

    CUDA_CALLABLE vector_int<vector_store_type, bits> get_left_shifted_vector(int shift) const {
        return get_shifted_vector<op_shift_left>(shift);
    }

    CUDA_CALLABLE vector_int<vector_store_type, bits> get_noted() const {
        return get_oped<op_not>();
    }

    template <int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_anded() const {
        return get_oped<op_and, constant>();
    }

    template <int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_ored() const {
        return get_oped<op_or, constant>();
    }

    template <int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_xored() const {
        return get_oped<op_xor, constant>();
    }

    template <int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_left_shifted_vector() const {
        return get_shifted_vector<op_shift_left, constant>();
    }

    template <int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_right_shifted_vector() const {
        return get_shifted_vector<op_shift_right, constant>();
    }

    template <typename tuple_of_pointers_storage_t>
    CUDA_CALLABLE static vector_int<vector_store_type, bits> load_from(
        tuple_of_pointers_storage_t storage, std::size_t offset) {

        vector_int<vector_store_type, bits> result;

        constexpr auto storage_size = std::tuple_size_v<tuple_of_pointers_storage_t>;
        constexpr auto loaded_bits = std::min<int>(storage_size, bits);

        for_each_in<loaded_bits>([&]<std::size_t bit_idx>() {
            auto ptr_to_ith_storage = std::get<bit_idx>(storage);
            std::get<bit_idx>(result.numbers) = ptr_to_ith_storage[offset];
        });

        return result;
    }

    static constexpr bool has_save_to_method = true;

    template <typename tuple_of_pointers_storage_t>
    CUDA_CALLABLE void save_to(tuple_of_pointers_storage_t storage, std::size_t offset) const {

        constexpr auto storage_size = std::tuple_size_v<tuple_of_pointers_storage_t>;
        constexpr auto saved_bits = std::min<int>(storage_size, bits);

        for_each_in<saved_bits>([&]<std::size_t bit_idx>() {
            auto ptr_to_ith_storage = std::get<bit_idx>(storage);
            ptr_to_ith_storage[offset] = std::get<bit_idx>(numbers);
        });
    }

    template <int other_bits>
    CUDA_CALLABLE vector_store_type equals_to(
        vector_int<vector_store_type, other_bits> other) const {

        constexpr int min_bits = (bits < other_bits ? bits : other_bits);
        vector_store_type result = constats_ops<vector_store_type>::ones;

        for_each_in<min_bits>([&]<std::size_t i>() {
            auto a = std::get<i>(numbers);
            auto b = std::get<i>(other.numbers);

            result &= ~(a ^ b);
        });

        if constexpr (other_bits < bits) {
            for_each_in<bits - other_bits>([&]<std::size_t i>() {
                auto a = std::get<min_bits + i>(numbers);
                result &= ~a;
            });
        }
        else if constexpr (other_bits > bits) {
            for_each_in<other_bits - bits>([&]<std::size_t i>() {
                auto a = std::get<min_bits + i>(other.numbers);
                result &= ~a;
            });
        }

        return result;
    }

    template <int other_bits>
    CUDA_CALLABLE vector_store_type not_equal_to(
        vector_int<vector_store_type, other_bits> other) const {

        constexpr int min_bits = (bits < other_bits ? bits : other_bits);
        vector_store_type result = 0;

        for_each_in<min_bits>([&]<std::size_t i>() {
            auto a = std::get<i>(numbers);
            auto b = std::get<i>(other.numbers);

            result |= a ^ b;
        });

        if constexpr (other_bits < bits) {
            for_each_in<bits - other_bits>([&]<std::size_t i>() {
                auto a = std::get<min_bits + i>(numbers);
                result |= a;
            });
        }
        else if constexpr (other_bits > bits) {
            for_each_in<other_bits - bits>([&]<std::size_t i>() {
                auto a = std::get<min_bits + i>(other.numbers);
                result |= a;
            });
        }

        return result;
    }

    template <int constant>
    CUDA_CALLABLE vector_store_type equals_to() const {
        // constexpr auto constant_bits
        //     = constats_ops<int>::get_highest_set_bit<constant>();

        vector_store_type result = constats_ops<vector_store_type>::ones;

        for_each_in<bits>([&]<std::size_t i>() {
            auto a = std::get<i>(numbers);

            constexpr auto constant_is_set =
                constats_ops<vector_store_type>::template is_set_at<constant>(i);

            if constexpr (constant_is_set) {
                result &= a;
            } else {
                result &= ~a;
            }
        });

        return result;
    }

    template <int other_bits>
    CUDA_CALLABLE vector_store_type greater_than(
        vector_int<vector_store_type, other_bits> other) const {

        vector_store_type result = 0;
        vector_store_type decided = 0;

        constexpr int min_bits = (bits < other_bits ? bits : other_bits);
        
        if constexpr (bits > other_bits) {
            for_each_in<bits - other_bits>([&]<std::size_t i_lower>() {
                constexpr std::size_t i = bits - i_lower - 1;
                auto a = std::get<i>(numbers);

                result = result | a;
            });
        }

        decided = result;
        
        if constexpr (other_bits > bits) {
            for_each_in<other_bits - bits>([&]<std::size_t i_lower>() {
                constexpr std::size_t i = other_bits - i_lower - 1;

                auto b = std::get<i>(other.numbers);
                decided = decided | b;
            });
        }

        for_each_in<min_bits>([&]<std::size_t i_lower>() {
            constexpr std::size_t i = min_bits - i_lower - 1;

            auto a = std::get<i>(numbers);
            auto b = std::get<i>(other.numbers);
            
            result = result | (~decided & ((a ^ b) & a));

            decided = decided | (a ^ b);
        });

        return result;
    }

  private:
    store_t numbers;

    template <typename Callback, std::size_t... Is>
    CUDA_CALLABLE static void for_each_bit_impl(Callback&& cb, std::index_sequence<Is...>) {
        (cb.template operator()<Is>(), ...);
    }

    template <typename Callback>
    CUDA_CALLABLE static void for_each_bit(Callback&& cb) {
        for_each_bit_impl(std::forward<Callback>(cb), std::make_index_sequence<bits>{});
    }

    template <int count, typename Callback>
    CUDA_CALLABLE static void for_each_in(Callback&& cb) {
        for_each_bit_impl(std::forward<Callback>(cb), std::make_index_sequence<count>{});
    }

    template <typename shift_op_t>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_shifted_vector(int shift) const {
        vector_int<vector_store_type, bits> result;

        for_each_in<bits>([&]<std::size_t i>() {
            auto word = std::get<i>(numbers);
            auto shifted_word = shift_op_t::apply(word, shift);

            std::get<i>(result.numbers) = shifted_word;
        });

        return result;
    }

    template <typename shift_op_t, int shift>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_shifted_vector() const {
        vector_int<vector_store_type, bits> result;

        for_each_in<bits>([&]<std::size_t i>() {
            auto word = std::get<i>(numbers);
            auto shifted_word = shift_op_t::template apply<shift>(word);

            std::get<i>(result.numbers) = shifted_word;
        });

        return result;
    }

    template <bit_action action>
    CUDA_CALLABLE vector_store_type apply_action(vector_store_type word) const {
        if constexpr (action == bit_action::SET_ZERO) {
            return 0;
        } else if constexpr (action == bit_action::SET_ONE) {
            return constats_ops<vector_store_type>::ones;
        } else if constexpr (action == bit_action::FLIP) {
            return ~word;
        } else if constexpr (action == bit_action::NO_ACTION) {
            return word;
        } else {
            static_assert("Unknown action for bit operation");
        }
    }

    template <int other_bits, typename op_t>
    CUDA_CALLABLE vector_int_higher_precision_t<other_bits> get_oped(
        vector_int<vector_store_type, other_bits> other) const {

        constexpr int min_bits = (bits < other_bits ? bits : other_bits);
        constexpr int max_bits = (bits > other_bits ? bits : other_bits);

        vector_int<vector_store_type, max_bits> result;

        for_each_in<min_bits>([&]<std::size_t i>() {
            auto a = std::get<i>(numbers);
            auto b = std::get<i>(other.numbers);

            std::get<i>(result.numbers) = op_t::apply(a, b);
        });

        if constexpr (other_bits < bits) {
            for_each_in<bits - min_bits>([&]<std::size_t i>() {
                constexpr auto action = op_t::template action_for_bit<int, 0>(0);

                auto a = std::get<min_bits + i>(numbers);
                std::get<min_bits + i>(result.numbers) = apply_action<action>(a);
            });
        } else if constexpr (other_bits > bits) {
            for_each_in<other_bits - min_bits>([&]<std::size_t i>() {
                constexpr auto action = op_t::template action_for_bit<int, 0>(0);

                auto a = std::get<min_bits + i>(other.numbers);
                std::get<min_bits + i>(result.numbers) = apply_action<action>(a);
            });
        }

        return result;
    }

    template <typename op_t>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_oped() const {
        vector_int<vector_store_type, bits> result;

        for_each_in<bits>([&]<std::size_t i>() {
            auto word = std::get<i>(numbers);
            auto oped_word = op_t::apply(word);

            std::get<i>(result.numbers) = oped_word;
        });

        return result;
    }

    template <typename op_t, int constant>
    CUDA_CALLABLE vector_int<vector_store_type, bits> get_oped() const {
        vector_int<vector_store_type, bits> result;

        for_each_in<bits>([&]<std::size_t i>() {
            constexpr auto action = op_t::template action_for_bit<int, constant>(i);

            auto word = std::get<i>(numbers);
            std::get<i>(result.numbers) = apply_action<action>(word);
        });

        return result;
    }
};

struct vector_int_factory {
    template <typename vector_store_type, int constant>
    CUDA_CALLABLE static auto from_constant() {
        constexpr auto bits
            = constats_ops<int>::get_highest_set_bit<constant>() + 1;

        if constexpr (bits == 0) {
            return vector_int<vector_store_type, 1>{};
        }

        vector_int<vector_store_type, bits + 1> result;

        for_each_in<bits + 1>([&]<std::size_t bit_idx>() {
            constexpr auto is_set
                = constats_ops<vector_store_type>::template is_set_at<constant>(bit_idx);

            if constexpr (is_set) {
                std::get<bit_idx>(result.numbers)
                    = constats_ops<vector_store_type>::ones;
            } else {
                std::get<bit_idx>(result.numbers) = 0;
            }
        });

        return result;
    }

    template <typename vector_store_type>
    CUDA_CALLABLE static auto from_condition_result(vector_store_type condition_result) {
        vector_int<vector_store_type, 1> result;
        std::get<0>(result.numbers) = condition_result;
        return result;
    }

    template <typename vector_store_type, typename pointer_storage_t>
    CUDA_CALLABLE static auto load_from(pointer_storage_t storage, std::size_t offset) {
        constexpr auto bits = std::tuple_size_v<pointer_storage_t>;
        static_assert(bits > 0, "Storage must have at least one element");

        return vector_int<vector_store_type, bits>::load_from(storage, offset);
    }

  private:
    template <typename Callback, std::size_t... Is>
    CUDA_CALLABLE static void for_each_bit_impl(Callback&& cb, std::index_sequence<Is...>) {
        (cb.template operator()<Is>(), ...);
    }

    template <int count, typename Callback>
    CUDA_CALLABLE static void for_each_in(Callback&& cb) {
        for_each_bit_impl(std::forward<Callback>(cb), std::make_index_sequence<count>{});
    }
};

}

#endif // CELLATO_CORE_VECTOR_INT_HPP