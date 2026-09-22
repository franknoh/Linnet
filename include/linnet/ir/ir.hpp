#pragma once

#include "linnet/sema/model.hpp"
#include "linnet/source/source_manager.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// Core Tensor IR: an SSA representation of checked programs.
//
//   Module -> Function -> Region -> Block -> Operation -> Value
//
// Every value has a semantic type from the analysis model. Functions keep the
// generic parameters of their source declaration; dimensions stay symbolic.
// Operations with regions (`if`, `option.match`, `comprehension`, `reduce`,
// `static_for`) evaluate their regions to the values their terminator yields.
//
// The IR is an implementation detail: its textual form (`inspect --core-ir`)
// is for debugging and is not stable.

namespace linnet::ir {

using ValueId = std::uint32_t;
using OpId = std::uint32_t;
using BlockId = std::uint32_t;
using RegionId = std::uint32_t;
using FunctionId = std::uint32_t;
inline constexpr std::uint32_t no_id = 0xFFFFFFFFU;

// X(EnumName, "spelling", operand count or -1 for variadic, region count)
#define LINNET_IR_OPS(X)                                                                           \
    X(ConstInt, "const.int", 0, 0)                                                                 \
    X(ConstFloat, "const.float", 0, 0)                                                             \
    X(ConstBool, "const.bool", 0, 0)                                                               \
    X(ConstDim, "const.dim", 0, 0)                                                                 \
    X(EnumConst, "enum.const", 0, 0)                                                               \
    X(Add, "add", 2, 0)                                                                            \
    X(Sub, "sub", 2, 0)                                                                            \
    X(Mul, "mul", 2, 0)                                                                            \
    X(Div, "div", 2, 0)                                                                            \
    X(Rem, "rem", 2, 0)                                                                            \
    X(Min, "min", 2, 0)                                                                            \
    X(Max, "max", 2, 0)                                                                            \
    X(Compare, "compare", 2, 0)                                                                    \
    X(And, "and", 2, 0)                                                                            \
    X(Or, "or", 2, 0)                                                                              \
    X(Not, "not", 1, 0)                                                                            \
    X(Neg, "neg", 1, 0)                                                                            \
    X(Exp, "exp", 1, 0)                                                                            \
    X(Log, "log", 1, 0)                                                                            \
    X(Sqrt, "sqrt", 1, 0)                                                                          \
    X(Rsqrt, "rsqrt", 1, 0)                                                                        \
    X(Sin, "sin", 1, 0)                                                                            \
    X(Cos, "cos", 1, 0)                                                                            \
    X(Tanh, "tanh", 1, 0)                                                                          \
    X(Abs, "abs", 1, 0)                                                                            \
    X(Cast, "cast", 1, 0)                                                                          \
    X(Select, "select", 3, 0)                                                                      \
    X(Reshape, "reshape", 1, 0)                                                                    \
    X(Permute, "permute", 1, 0)                                                                    \
    X(Broadcast, "broadcast", 1, 0)                                                                \
    X(Slice, "slice", 1, 0)                                                                        \
    X(Concat, "concat", -1, 0)                                                                     \
    X(Fill, "fill", 1, 0)                                                                          \
    X(Iota, "iota", 0, 0)                                                                          \
    X(Element, "tensor.element", -1, 0)                                                            \
    X(Comprehension, "comprehension", 0, 1)                                                        \
    X(Reduce, "reduce", 0, 1)                                                                      \
    X(TupleMake, "tuple.make", -1, 0)                                                              \
    X(TupleGet, "tuple.get", 1, 0)                                                                 \
    X(StructMake, "struct.make", -1, 0)                                                            \
    X(StructGet, "struct.get", 1, 0)                                                               \
    X(OptionSome, "option.some", 1, 0)                                                             \
    X(OptionNone, "option.none", 0, 0)                                                             \
    X(OptionMatch, "option.match", 1, 2)                                                           \
    X(EnumMatch, "enum.match", 1, -1)                                                              \
    X(If, "if", 1, 2)                                                                              \
    X(Call, "call", -1, 0)                                                                         \
    X(SemanticCall, "semantic.call", -1, 0)                                                        \
    X(BlockParam, "block.param", 1, 0)                                                             \
    X(BlockSub, "block.sub", 1, 0)                                                                 \
    X(ArrayGet, "array.get", 2, 0)                                                                 \
    X(StaticFor, "static_for", -1, 1)                                                              \
    X(Yield, "yield", -1, 0)                                                                       \
    X(Return, "return", -1, 0)

enum class OpKind : std::uint8_t {
#define LINNET_IR_ENUM(name, spelling, operands, regions) name,
    LINNET_IR_OPS(LINNET_IR_ENUM)
#undef LINNET_IR_ENUM
};

std::string_view op_spelling(OpKind kind);
int op_operand_count(OpKind kind); // -1 when variadic
int op_region_count(OpKind kind);  // -1 when variadic

enum class CompareKind : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };
enum class ReduceKind : std::uint8_t { Sum, Prod, Max, Min, Any, All };

std::string_view compare_spelling(CompareKind kind);
std::string_view reduce_spelling(ReduceKind kind);

// Attributes an operation may carry; which ones matter depends on the kind.
struct Attributes {
    std::int64_t integer = 0; // const.int, const.bool (0/1), tuple.get, struct.get, array index
    double number = 0.0;      // const.float
    shape::Poly dim;          // const.dim
    std::string name;         // call/semantic.call callee id; block.param member;
                              // enum.const variant; comprehension/reduce index names
    std::vector<std::string> names; // enum.match variants; comprehension/reduce index names
    CompareKind compare = CompareKind::Eq;
    ReduceKind reduce = ReduceKind::Sum;
    sema::Shape shape;               // reshape/broadcast/fill target; permute axes as constants;
                                     // comprehension/reduce index domains (one unit per index)
    std::vector<shape::Poly> starts; // slice
    std::vector<shape::Poly> stops;
    std::vector<std::int64_t> steps;
    std::vector<bool> squeezed;      // slice: axes removed by an integer index
    std::vector<bool> whole;         // slice: axes kept entirely (shape packs)
    sema::Shape pack_units;          // slice: the shape unit of each axis (the pack for whole axes)
    std::int64_t axis = 0;           // concat
    sema::Substitution substitution; // call: generic bindings
    std::vector<sema::GenericValue> generic_args; // call: the same, in the callee's order

    friend bool operator==(const Attributes&, const Attributes&) = default;
};

struct Value {
    sema::TypeId type = sema::no_type;
    OpId producer = no_id; // no_id for block arguments
    BlockId block = no_id; // block that owns it (as argument or via producer)
    std::string name;      // source name when there is one, for readability
};

struct Operation {
    OpKind kind = OpKind::Return;
    std::vector<ValueId> operands;
    std::vector<ValueId> results;
    std::vector<RegionId> regions;
    Attributes attributes;
    SourceSpan span;
    BlockId block = no_id;
};

struct Block {
    std::vector<ValueId> arguments;
    std::vector<OpId> ops;
    RegionId region = no_id;
};

struct Region {
    std::vector<BlockId> blocks; // one block per region so far
    OpId parent = no_id;         // no_id for a function body
};

struct Function {
    std::string name; // semantic identity: `module.path::name` (`Block.method` for methods)
    bool is_op = false;
    bool is_entry = false;
    sema::EntityId entity = sema::no_entity;
    std::vector<sema::GenericInfo> generics;
    std::vector<sema::ConstraintInfo> constraints;
    RegionId body = no_id;
    std::vector<sema::TypeId> results;
};

class Module {
public:
    explicit Module(std::shared_ptr<sema::Model> model) : model_(std::move(model)) {}

    sema::Model& model() { return *model_; }
    const sema::Model& model() const { return *model_; }
    sema::TypeStore& types() { return model_->types; }
    const sema::TypeStore& types() const { return model_->types; }

    const std::vector<Function>& functions() const { return functions_; }
    Function& function(FunctionId id) { return functions_[id]; }
    const Function& function(FunctionId id) const { return functions_[id]; }
    const Value& value(ValueId id) const { return values_[id]; }
    Value& value(ValueId id) { return values_[id]; }
    const Operation& op(OpId id) const { return ops_[id]; }
    Operation& op(OpId id) { return ops_[id]; }
    const Block& block(BlockId id) const { return blocks_[id]; }
    Block& block(BlockId id) { return blocks_[id]; }
    const Region& region(RegionId id) const { return regions_[id]; }
    Region& region(RegionId id) { return regions_[id]; }
    std::size_t value_count() const { return values_.size(); }

    FunctionId add_function(Function function);
    RegionId add_region(OpId parent);
    BlockId add_block(RegionId region);
    ValueId add_argument(BlockId block, sema::TypeId type, std::string name = {});
    // Rewrites every use of `from` in the module to `to`.
    void replace_uses(ValueId from, ValueId to);
    // Removes `id` from its block. Its results must have no uses.
    void erase_op(OpId id);
    // Every block of the module, outermost first.
    std::vector<BlockId> all_blocks() const;

    // Appends an operation to `block` and creates one value per result type.
    OpId add_op(BlockId block,
                OpKind kind,
                std::vector<ValueId> operands,
                const std::vector<sema::TypeId>& result_types,
                Attributes attributes = {},
                SourceSpan span = {});

private:
    std::shared_ptr<sema::Model> model_;
    std::vector<Function> functions_;
    std::deque<Value> values_;
    std::deque<Operation> ops_;
    std::deque<Block> blocks_;
    std::deque<Region> regions_;
};

// Human-readable dump of the whole module.
std::string print(const Module& module);

// Structural checks: operand counts, region counts, dominance within a block,
// terminators, and result types of the operations whose types are fixed by
// their operands. Returns problems as text; empty when the module is valid.
std::vector<std::string> verify(const Module& module);

} // namespace linnet::ir
