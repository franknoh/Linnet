#pragma once

#include "linnet/shape/dim.hpp"
#include "linnet/shape/solver.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Semantic types. Types are immutable values stored in a TypeStore and
// referred to by id. Dimension equality is a proof obligation rather than a
// structural property, so types are compared through `TypeStore::equal` with
// the solver of the function being checked, never by id.

namespace linnet::sema {

enum class ScalarKind : std::uint8_t {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    // Storage formats: cast to and from, moved and gathered, but computed
    // with only after a cast (spec §3.1).
    F8E4M3, // float(exponent: 4, mantissa: 3, inf: false): FP8 E4M3FN
    F8E5M2, // float(exponent: 5, mantissa: 2): FP8 E5M2
    // Packed storage formats: two elements a byte along the last axis.
    I4,    // int(bits: 4)
    U4,    // int(bits: 4, signed: false)
    F4E2M1 // float(exponent: 2, mantissa: 1, inf: false, nan: false): FP4 E2M1
};

// What a dtype is, as a `dtype` declaration spells it (spec §3.1): its
// family, and for an integer its width and sign, for a float its exponent
// and mantissa widths and whether it has infinities and NaNs.
struct DTypeFormat {
    enum class Family : std::uint8_t { Bool, Int, Float };
    Family family = Family::Bool;
    std::uint32_t bits = 8;
    bool is_signed = true;
    std::uint32_t exponent = 0;
    std::uint32_t mantissa = 0;
    bool inf = true;
    bool nan = true;
    friend bool operator==(const DTypeFormat&, const DTypeFormat&) = default;
};

DTypeFormat format_of(ScalarKind kind);
// The scalar kind with `format`, if the compiler knows one.
std::optional<ScalarKind> scalar_of(const DTypeFormat& format);
// The format a `dtype` declaration describes: a `float` (else an `int`) with
// its parameters by name, those left out at their defaults.
DTypeFormat dtype_format(bool is_float, const std::map<std::string_view, std::int64_t>& values);

// Every scalar kind's canonical name, as plans, bindings and generated code
// spell it; and back.
std::string_view scalar_name(ScalarKind kind);

// The smallest and largest value of an integer dtype, the largest unsigned so
// that u64's fits (bool: 0 and 1).
struct IntegerLimits {
    std::int64_t lowest = 0;
    std::uint64_t highest = 0;
};
IntegerLimits integer_limits(ScalarKind kind);
std::optional<ScalarKind> scalar_from_name(std::string_view name);
// The names source can use without declaring them: `bool`, and those the
// prelude (`stdlib/prelude.linnet`, compiled in) declares: `i8` to `i64`,
// `u8` to `u64`, `f16`, `bf16`, `f32`, `f64`. Others come from `dtype`
// declarations (`std.dtypes`).
std::optional<ScalarKind> builtin_scalar(std::string_view name);
// The dtypes the prelude declares, by name.
const std::map<std::string, ScalarKind, std::less<>>& prelude_dtypes();
// Floats and integers that compute; a storage format is neither.
bool is_float(ScalarKind kind);
bool is_integer(ScalarKind kind);
bool is_storage(ScalarKind kind);
// A storage format narrower than a byte, stored two elements a byte along a
// tensor's last axis (spec §3.1).
bool is_packed(ScalarKind kind);

// The generic constraints that classify element types.
enum class DTypeClass : std::uint8_t { Any, Numeric, Integer, Float };

bool class_contains(DTypeClass outer, DTypeClass inner);
// A dtype class as source bounds a generic with it (`DType`, `Float`), as
// plans spell it (`any`, `float`), and back from a plan.
std::string_view dtype_class_name(DTypeClass dtype_class);
std::string_view dtype_class_key(DTypeClass dtype_class);
std::optional<DTypeClass> parse_dtype_class_key(std::string_view key);

using DTypeVarId = std::uint32_t;

// An element type: a concrete scalar kind or a generic `T: Float` variable.
// A codebook format (`codebook`, from 1; spec §3.1) is stored as codes of
// its carrier, `scalar`, but is a type of its own.
struct DType {
    bool is_var = false;
    ScalarKind scalar = ScalarKind::F32;
    DTypeVarId var = 0;
    std::uint32_t codebook = 0;

    static DType of(ScalarKind kind) { return {false, kind, 0, 0}; }
    static DType variable(DTypeVarId id) { return {true, ScalarKind::F32, id, 0}; }
    // Held as it is but computed with only after a cast: a storage format
    // or a codebook.
    bool stores_only() const { return !is_var && (codebook != 0 || is_storage(scalar)); }
    friend bool operator==(const DType&, const DType&) = default;
};

// A codebook format, `dtype nf4 = codebook(carrier: u4, values: [...])`: each
// code of its carrier (an unsigned integer format) stands for one of
// `values`, which increase.
struct Codebook {
    std::string name;
    ScalarKind carrier = ScalarKind::U8;
    std::vector<double> values;
};

// One unit of a shape: a dimension, or a whole shape pack such as `*S`.
struct ShapeElem {
    bool is_pack = false;
    shape::SymbolId pack = 0;
    shape::Poly dim;

    static ShapeElem of(shape::Poly dim) { return {false, 0, std::move(dim)}; }
    static ShapeElem of_pack(shape::SymbolId pack) { return {true, pack, {}}; }
    // Structural identity, not provable equality.
    friend bool operator==(const ShapeElem&, const ShapeElem&) = default;
};
using Shape = std::vector<ShapeElem>;

using TypeId = std::uint32_t;
using EntityId = std::uint32_t;
inline constexpr std::uint32_t no_type = 0xFFFFFFFFU;

enum class TypeKind : std::uint8_t {
    Error, // already diagnosed; compatible with everything to avoid cascades
    Unit,  // result of a function without a declared result
    Scalar,
    Tensor,
    Tuple,
    Optional,
    Array, // compile-time structural array `[T; N]`
    Struct,
    Enum,
    Block,
    // Compile-time and contextual values.
    CompileInt,   // integer literal, generic dimension, or constant arithmetic
    FloatLiteral, // float literal awaiting a dtype from context
    NoneLiteral,  // `none` awaiting an optional type from context
    ShapeValue,   // `[B, S, H]`
};

// A generic argument: what a `Dim`, `Shape`, or dtype parameter is bound to.
struct GenericValue {
    enum class Kind : std::uint8_t { Dim, Pack, DType } kind = Kind::Dim;
    shape::Poly dim;
    Shape shape;
    DType dtype;
    friend bool operator==(const GenericValue&, const GenericValue&) = default;
};

struct TypeData {
    TypeKind kind = TypeKind::Error;
    DType dtype;                  // Scalar, Tensor
    Shape shape;                  // Tensor, ShapeValue
    std::vector<TypeId> elements; // Tuple; Optional and Array hold one
    shape::Poly value;            // CompileInt value; Array length
    std::optional<double> number; // FloatLiteral, when known
    EntityId decl = 0;            // Struct, Enum, Block
    std::vector<GenericValue> args;
};

// Simultaneous replacement of generic parameters.
struct Substitution {
    std::map<shape::SymbolId, shape::Poly> dims;
    std::map<shape::SymbolId, Shape> packs;
    std::map<DTypeVarId, DType> dtypes;

    bool empty() const { return dims.empty() && packs.empty() && dtypes.empty(); }
    friend bool operator==(const Substitution&, const Substitution&) = default;
};

struct DTypeVar {
    std::string name;
    DTypeClass constraint;
};

class TypeStore {
public:
    explicit TypeStore(shape::DimContext& dims);

    shape::DimContext& dims() { return *dims_; }
    const TypeData& get(TypeId id) const { return types_[id]; }
    TypeKind kind(TypeId id) const { return types_[id].kind; }
    bool is_error(TypeId id) const { return types_[id].kind == TypeKind::Error; }

    TypeId error() const { return 0; }
    TypeId unit() const { return 1; }
    TypeId add(TypeData data);
    TypeId scalar(DType dtype);
    TypeId scalar(ScalarKind kind) { return scalar(DType::of(kind)); }
    TypeId tensor(Shape shape, DType dtype);
    TypeId tuple(std::vector<TypeId> elements);
    TypeId optional(TypeId inner);
    TypeId array(TypeId element, shape::Poly length);
    TypeId compile_int(shape::Poly value);
    TypeId float_literal(std::optional<double> value);
    TypeId none_literal();
    TypeId shape_value(Shape shape);
    TypeId nominal(TypeKind kind, EntityId decl, std::vector<GenericValue> args);

    DTypeVarId add_dtype_var(std::string name, DTypeClass constraint);
    const DTypeVar& dtype_var(DTypeVarId id) const { return dtype_vars_[id]; }
    DTypeClass class_of(DType dtype) const;
    // A codebook format's dtype, and the codebook a dtype is (or null).
    DType add_codebook(Codebook codebook);
    const Codebook* codebook_of(DType dtype) const;

    // Element count of a shape; packs contribute their symbolic size.
    shape::Poly element_count(const Shape& shape);

    Shape substitute(const Shape& shape, const Substitution& substitution);
    shape::Poly substitute(const shape::Poly& poly, const Substitution& substitution);
    DType substitute(DType dtype, const Substitution& substitution) const;
    GenericValue substitute(const GenericValue& value, const Substitution& substitution);
    TypeId substitute(TypeId type, const Substitution& substitution);
    // `inner` with every binding substituted by `outer`: a call's bindings
    // in terms of its caller's.
    Substitution substitute(Substitution inner, const Substitution& outer);

    bool equal(const Shape& a, const Shape& b, shape::Solver& solver);
    bool equal(const GenericValue& a, const GenericValue& b, shape::Solver& solver);
    bool equal(TypeId a, TypeId b, shape::Solver& solver);

    std::string to_string(DType dtype) const;
    std::string to_string(const Shape& shape) const; // `B, S, H` without brackets
    std::string to_string(const GenericValue& value) const;
    std::string to_string(TypeId type) const;

    // Supplies names for struct, enum, and block declarations in to_string.
    void set_entity_namer(std::function<std::string(EntityId)> namer) {
        entity_namer_ = std::move(namer);
    }

private:
    shape::DimContext* dims_;
    std::deque<TypeData> types_; // deque: references stay valid as types are added
    std::vector<DTypeVar> dtype_vars_;
    std::vector<Codebook> codebooks_;
    std::function<std::string(EntityId)> entity_namer_;
};

} // namespace linnet::sema
