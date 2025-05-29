#ifndef CELLATO_AST_HPP
#define CELLATO_AST_HPP

namespace cellato::ast {

template <auto Value>
struct constant {
    using type = decltype(Value);
    static constexpr type value = Value;
};

template <auto Value>
struct state_constant {
    using type = decltype(Value);
    static constexpr type value = Value;
};

template <int x_offset_val, int y_offset_val>
struct neighbor_at {
    static constexpr int x_offset = x_offset_val;
    static constexpr int y_offset = y_offset_val;
};

struct moore_8_neighbors {};
struct von_neumann_4_neighbors {};

template <typename Constant, typename Neighborhood>
struct count_neighbors {
    using state_constant = Constant;
    using neighborhood = Neighborhood;
};

template <typename Condition, typename Then, typename Else>
struct if_then_else {
    using condition = Condition;
    using then_expr = Then;
    using else_expr = Else;
};

template <typename Left, typename Right>
struct and_ {
    using left = Left;
    using right = Right;
};

template <typename Left, typename Right>
struct or_ {
    using left = Left;
    using right = Right;
};

template <typename Left, typename Right>
struct equals {
    using left = Left;
    using right = Right;
};

template <typename Left, typename Right>
struct greater_than {
    using left = Left;
    using right = Right;
};

template <typename Left, typename Right>
struct not_equals {
    using left = Left;
    using right = Right;
};

template <typename... Args>
class __unpacked_if;

template <typename E>
class __unpacked_if<E> {
public:
    using nested_if_then_else = E;
};

template <typename C, typename T, typename... Chain>
class __unpacked_if<C, T, Chain...> {
public:
    using nested_if_then_else = if_then_else<
        C, T,
        typename __unpacked_if<Chain...>::nested_if_then_else>;
};

template <typename Condition, typename ...ChainOfThenElse>
struct if_ {
    template <typename Then>
    struct then_ {
        template <typename Else>
        using else_ = typename __unpacked_if<Condition, ChainOfThenElse..., Then, Else>::nested_if_then_else;

        template <typename ElseCondition>
        using elif_ = if_<Condition, ChainOfThenElse..., Then, ElseCondition>;
    };
};


template <typename Left, template <typename, typename> class Operator, typename Right>
using p = Operator<Left, Right>;

using current_state = neighbor_at<0, 0>;

}

#endif // CELLATO_AST_HPP