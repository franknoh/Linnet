#include "linnet/sema/types.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace linnet::sema {

namespace {

constexpr auto scalar_names = std::to_array<std::string_view>(
    {"bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "f64"});

// In `DTypeClass` order.
constexpr auto class_names =
    std::to_array<std::string_view>({"DType", "Numeric", "Integer", "Float"});
constexpr auto class_keys = std::to_array<std::string_view>({"any", "numeric", "integer", "float"});

} // namespace

std::string_view dtype_class_name(DTypeClass dtype_class) {
    return class_names[static_cast<std::size_t>(dtype_class)];
}

std::string_view dtype_class_key(DTypeClass dtype_class) {
    return class_keys[static_cast<std::size_t>(dtype_class)];
}

std::optional<DTypeClass> parse_dtype_class_key(std::string_view key) {
    for (std::size_t i = 0; i < class_keys.size(); ++i) {
        if (class_keys[i] == key) {
            return static_cast<DTypeClass>(i);
        }
    }
    return std::nullopt;
}

IntegerLimits integer_limits(ScalarKind kind) {
    const auto of = [](auto type) {
        using T = decltype(type);
        return IntegerLimits{static_cast<std::int64_t>(std::numeric_limits<T>::min()),
                             static_cast<std::uint64_t>(std::numeric_limits<T>::max())};
    };
    switch (kind) {
    case ScalarKind::Bool:
        return {0, 1};
    case ScalarKind::I8:
        return of(std::int8_t{});
    case ScalarKind::I16:
        return of(std::int16_t{});
    case ScalarKind::I32:
        return of(std::int32_t{});
    case ScalarKind::I64:
        return of(std::int64_t{});
    case ScalarKind::U8:
        return of(std::uint8_t{});
    case ScalarKind::U16:
        return of(std::uint16_t{});
    case ScalarKind::U32:
        return of(std::uint32_t{});
    case ScalarKind::U64:
        return of(std::uint64_t{});
    default:
        return {};
    }
}

std::string_view scalar_name(ScalarKind kind) {
    return scalar_names[static_cast<std::size_t>(kind)];
}

std::optional<ScalarKind> scalar_from_name(std::string_view name) {
    for (std::size_t i = 0; i < scalar_names.size(); ++i) {
        if (scalar_names[i] == name) {
            return static_cast<ScalarKind>(i);
        }
    }
    return std::nullopt;
}

bool is_float(ScalarKind kind) {
    return kind >= ScalarKind::F16;
}

bool is_integer(ScalarKind kind) {
    return kind >= ScalarKind::I8 && kind <= ScalarKind::U64;
}

bool class_contains(DTypeClass outer, ScalarKind kind) {
    switch (outer) {
    case DTypeClass::Any:
        return true;
    case DTypeClass::Numeric:
        return kind != ScalarKind::Bool;
    case DTypeClass::Integer:
        return is_integer(kind);
    case DTypeClass::Float:
        return is_float(kind);
    }
    return false;
}

bool class_contains(DTypeClass outer, DTypeClass inner) {
    if (outer == DTypeClass::Any || outer == inner) {
        return true;
    }
    return outer == DTypeClass::Numeric &&
           (inner == DTypeClass::Integer || inner == DTypeClass::Float);
}

// ------------------------------------------------------------------- creation

TypeStore::TypeStore(shape::DimContext& dims) : dims_(&dims) {
    types_.push_back({TypeKind::Error, {}, {}, {}, {}, {}, 0, {}});
    types_.push_back({TypeKind::Unit, {}, {}, {}, {}, {}, 0, {}});
}

TypeId TypeStore::add(TypeData data) {
    types_.push_back(std::move(data));
    return static_cast<TypeId>(types_.size() - 1);
}

TypeId TypeStore::scalar(DType dtype) {
    TypeData data;
    data.kind = TypeKind::Scalar;
    data.dtype = dtype;
    return add(std::move(data));
}

TypeId TypeStore::tensor(Shape shape, DType dtype) {
    TypeData data;
    data.kind = TypeKind::Tensor;
    data.dtype = dtype;
    data.shape = std::move(shape);
    return add(std::move(data));
}

TypeId TypeStore::tuple(std::vector<TypeId> elements) {
    TypeData data;
    data.kind = TypeKind::Tuple;
    data.elements = std::move(elements);
    return add(std::move(data));
}

TypeId TypeStore::optional(TypeId inner) {
    TypeData data;
    data.kind = TypeKind::Optional;
    data.elements = {inner};
    return add(std::move(data));
}

TypeId TypeStore::array(TypeId element, shape::Poly length) {
    TypeData data;
    data.kind = TypeKind::Array;
    data.elements = {element};
    data.value = std::move(length);
    return add(std::move(data));
}

TypeId TypeStore::compile_int(shape::Poly value) {
    TypeData data;
    data.kind = TypeKind::CompileInt;
    data.value = std::move(value);
    return add(std::move(data));
}

TypeId TypeStore::float_literal(std::optional<double> value) {
    TypeData data;
    data.kind = TypeKind::FloatLiteral;
    data.number = value;
    return add(std::move(data));
}

TypeId TypeStore::none_literal() {
    TypeData data;
    data.kind = TypeKind::NoneLiteral;
    return add(std::move(data));
}

TypeId TypeStore::shape_value(Shape shape) {
    TypeData data;
    data.kind = TypeKind::ShapeValue;
    data.shape = std::move(shape);
    return add(std::move(data));
}

TypeId TypeStore::nominal(TypeKind kind, EntityId decl, std::vector<GenericValue> args) {
    TypeData data;
    data.kind = kind;
    data.decl = decl;
    data.args = std::move(args);
    return add(std::move(data));
}

DTypeVarId TypeStore::add_dtype_var(std::string name, DTypeClass constraint) {
    dtype_vars_.push_back({std::move(name), constraint});
    return static_cast<DTypeVarId>(dtype_vars_.size() - 1);
}

DTypeClass TypeStore::class_of(DType dtype) const {
    if (dtype.is_var) {
        return dtype_vars_[dtype.var].constraint;
    }
    if (is_float(dtype.scalar)) {
        return DTypeClass::Float;
    }
    return is_integer(dtype.scalar) ? DTypeClass::Integer : DTypeClass::Any;
}

shape::Poly TypeStore::element_count(const Shape& shape) {
    shape::Poly count(1);
    for (const ShapeElem& element : shape) {
        count = count * (element.is_pack ? dims_->pack_size(element.pack) : element.dim);
    }
    return count;
}

// --------------------------------------------------------------- substitution

shape::Poly TypeStore::substitute(const shape::Poly& poly, const Substitution& substitution) {
    if (substitution.dims.empty() && substitution.packs.empty()) {
        return poly;
    }
    shape::DimContext::Substitution lowered;
    lowered.dims = substitution.dims;
    for (const auto& [pack, shape] : substitution.packs) {
        lowered.pack_sizes[pack] = element_count(shape);
    }
    return dims_->substitute(poly, lowered);
}

Shape TypeStore::substitute(const Shape& shape, const Substitution& substitution) {
    Shape result;
    for (const ShapeElem& element : shape) {
        if (!element.is_pack) {
            result.push_back(ShapeElem::of(substitute(element.dim, substitution)));
        } else if (const auto found = substitution.packs.find(element.pack);
                   found != substitution.packs.end()) {
            result.insert(result.end(), found->second.begin(), found->second.end());
        } else {
            result.push_back(element);
        }
    }
    return result;
}

DType TypeStore::substitute(DType dtype, const Substitution& substitution) const {
    if (dtype.is_var) {
        if (const auto found = substitution.dtypes.find(dtype.var);
            found != substitution.dtypes.end()) {
            return found->second;
        }
    }
    return dtype;
}

GenericValue TypeStore::substitute(const GenericValue& value, const Substitution& substitution) {
    GenericValue result = value;
    result.dim = substitute(value.dim, substitution);
    result.shape = substitute(value.shape, substitution);
    result.dtype = substitute(value.dtype, substitution);
    return result;
}

Substitution TypeStore::substitute(Substitution inner, const Substitution& outer) {
    for (auto& [symbol, dim] : inner.dims) {
        dim = substitute(dim, outer);
    }
    for (auto& [symbol, shape] : inner.packs) {
        shape = substitute(shape, outer);
    }
    for (auto& [var, dtype] : inner.dtypes) {
        dtype = substitute(dtype, outer);
    }
    return inner;
}

TypeId TypeStore::substitute(TypeId type, const Substitution& substitution) {
    if (substitution.empty()) {
        return type;
    }
    TypeData data = get(type); // copy: `add` below may reallocate
    switch (data.kind) {
    case TypeKind::Error:
    case TypeKind::Unit:
    case TypeKind::FloatLiteral:
    case TypeKind::NoneLiteral:
        return type;
    default:
        break;
    }
    data.dtype = substitute(data.dtype, substitution);
    data.shape = substitute(data.shape, substitution);
    data.value = substitute(data.value, substitution);
    for (TypeId& element : data.elements) {
        element = substitute(element, substitution);
    }
    for (GenericValue& arg : data.args) {
        arg = substitute(arg, substitution);
    }
    return add(std::move(data));
}

// ------------------------------------------------------------------- equality

bool TypeStore::equal(const Shape& a, const Shape& b, shape::Solver& solver) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].is_pack != b[i].is_pack) {
            return false;
        }
        if (a[i].is_pack ? a[i].pack != b[i].pack : !solver.prove_equal(a[i].dim, b[i].dim)) {
            return false;
        }
    }
    return true;
}

bool TypeStore::equal(const GenericValue& a, const GenericValue& b, shape::Solver& solver) {
    if (a.kind != b.kind) {
        return false;
    }
    switch (a.kind) {
    case GenericValue::Kind::Dim:
        return solver.prove_equal(a.dim, b.dim);
    case GenericValue::Kind::Pack:
        return equal(a.shape, b.shape, solver);
    case GenericValue::Kind::DType:
        return a.dtype == b.dtype;
    }
    return false;
}

bool TypeStore::equal(TypeId a, TypeId b, shape::Solver& solver) {
    const TypeData& x = get(a);
    const TypeData& y = get(b);
    if (x.kind == TypeKind::Error || y.kind == TypeKind::Error) {
        return true;
    }
    if (x.kind != y.kind) {
        return false;
    }
    switch (x.kind) {
    case TypeKind::Error:
    case TypeKind::Unit:
    case TypeKind::NoneLiteral:
    case TypeKind::FloatLiteral:
        return true;
    case TypeKind::Scalar:
        return x.dtype == y.dtype;
    case TypeKind::Tensor:
        return x.dtype == y.dtype && equal(x.shape, y.shape, solver);
    case TypeKind::ShapeValue:
        return equal(x.shape, y.shape, solver);
    case TypeKind::CompileInt:
        return solver.prove_equal(x.value, y.value);
    case TypeKind::Array:
        return solver.prove_equal(x.value, y.value) &&
               equal(x.elements.front(), y.elements.front(), solver);
    case TypeKind::Tuple:
    case TypeKind::Optional:
        if (x.elements.size() != y.elements.size()) {
            return false;
        }
        for (std::size_t i = 0; i < x.elements.size(); ++i) {
            if (!equal(x.elements[i], y.elements[i], solver)) {
                return false;
            }
        }
        return true;
    case TypeKind::Struct:
    case TypeKind::Enum:
    case TypeKind::Block:
        if (x.decl != y.decl || x.args.size() != y.args.size()) {
            return false;
        }
        for (std::size_t i = 0; i < x.args.size(); ++i) {
            if (!equal(x.args[i], y.args[i], solver)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// ------------------------------------------------------------------ rendering

std::string TypeStore::to_string(DType dtype) const {
    return dtype.is_var ? dtype_vars_[dtype.var].name : std::string(scalar_name(dtype.scalar));
}

std::string TypeStore::to_string(const Shape& shape) const {
    std::string text;
    for (const ShapeElem& element : shape) {
        text += text.empty() ? "" : ", ";
        text += element.is_pack ? "*" + std::string(dims_->symbol_name(element.pack))
                                : dims_->to_string(element.dim);
    }
    return text;
}

std::string TypeStore::to_string(const GenericValue& value) const {
    switch (value.kind) {
    case GenericValue::Kind::Dim:
        return dims_->to_string(value.dim);
    case GenericValue::Kind::Pack:
        return "[" + to_string(value.shape) + "]";
    case GenericValue::Kind::DType:
        return to_string(value.dtype);
    }
    return "?";
}

std::string TypeStore::to_string(TypeId type) const {
    const TypeData& data = get(type);
    switch (data.kind) {
    case TypeKind::Error:
        return "<error>";
    case TypeKind::Unit:
        return "()";
    case TypeKind::Scalar:
        return to_string(data.dtype);
    case TypeKind::Tensor:
        return "Tensor[" + to_string(data.shape) + "; " + to_string(data.dtype) + "]";
    case TypeKind::Tuple: {
        std::string text = "(";
        for (const TypeId element : data.elements) {
            text += text.size() == 1 ? "" : ", ";
            text += to_string(element);
        }
        return text + ")";
    }
    case TypeKind::Optional:
        return to_string(data.elements.front()) + "?";
    case TypeKind::Array:
        return "[" + to_string(data.elements.front()) + "; " + dims_->to_string(data.value) + "]";
    case TypeKind::Struct:
    case TypeKind::Enum:
    case TypeKind::Block: {
        std::string text = entity_namer_ ? entity_namer_(data.decl) : std::string("<nominal>");
        if (!data.args.empty()) {
            text += "<";
            for (std::size_t i = 0; i < data.args.size(); ++i) {
                text += i == 0 ? "" : ", ";
                text += to_string(data.args[i]);
            }
            text += ">";
        }
        return text;
    }
    case TypeKind::CompileInt:
        return data.value.constant() ? "integer literal"
                                     : "dimension " + dims_->to_string(data.value);
    case TypeKind::FloatLiteral:
        return "float literal";
    case TypeKind::NoneLiteral:
        return "none";
    case TypeKind::ShapeValue:
        return "shape [" + to_string(data.shape) + "]";
    }
    return "?";
}

} // namespace linnet::sema
