#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// Symbolic dimension expressions.
//
// A dimension expression is kept in a canonical form: a polynomial with
// integer coefficients over *atoms*. An atom is a symbol or an operation that
// is not polynomial (floor division, remainder, min, max, the element count of
// a shape pack) whose operands are themselves canonical polynomials. Two
// expressions that are equal as polynomials therefore compare equal
// structurally, which makes `B * S * H`, `H * (S * B)` and `S * B * H + 0`
// indistinguishable to the type checker.
//
// Normalization here never uses assumptions. Reasoning from `where`
// constraints lives in shape::Solver.

namespace linnet::shape {

using SymbolId = std::uint32_t;
using AtomId = std::uint32_t;

enum class SymbolKind : std::uint8_t { Dim, Pack };

// One product of atoms with a coefficient. `atoms` is sorted; a repeated id
// denotes a power.
struct Term {
    std::vector<AtomId> atoms;
    std::int64_t coefficient = 0;

    friend auto operator<=>(const Term&, const Term&) = default;
    friend bool operator==(const Term&, const Term&) = default;
};

class DimContext;

// Canonical polynomial: terms sorted by their atom lists, no zero
// coefficients. The zero polynomial has no terms. A polynomial becomes
// invalid when its arithmetic overflows or divides by a constant zero;
// invalidity is sticky through every operation.
class Poly {
public:
    Poly() = default;
    explicit Poly(std::int64_t constant);

    static Poly invalid();
    static Poly atom(AtomId id);

    bool is_valid() const { return is_valid_; }
    bool is_zero() const { return is_valid_ && terms_.empty(); }
    const std::vector<Term>& terms() const { return terms_; }

    // The value when the polynomial is a constant.
    std::optional<std::int64_t> constant() const;
    std::int64_t constant_term() const;
    // The atom when the polynomial is exactly one atom with coefficient 1.
    std::optional<AtomId> single_atom() const;
    bool is_single_term() const { return is_valid_ && terms_.size() == 1; }

    friend Poly operator+(const Poly& a, const Poly& b);
    friend Poly operator-(const Poly& a, const Poly& b);
    friend Poly operator*(const Poly& a, const Poly& b);
    Poly operator-() const;

    // Exact quotient when `divisor` is a single term that divides every term.
    std::optional<Poly> exact_div(const Poly& divisor) const;

    friend auto operator<=>(const Poly&, const Poly&) = default;
    friend bool operator==(const Poly&, const Poly&) = default;

private:
    friend class DimContext;
    void normalize();

    std::vector<Term> terms_;
    bool is_valid_ = true;
};

enum class AtomKind : std::uint8_t { Symbol, FloorDiv, Mod, Min, Max, PackSize };

struct Atom {
    AtomKind kind = AtomKind::Symbol;
    SymbolId symbol = 0; // Symbol, PackSize
    Poly lhs;            // binary kinds
    Poly rhs;
};

// Owns symbols and interned atoms. Atom ids follow creation order, so
// canonical forms are deterministic for a given sequence of declarations.
class DimContext {
public:
    SymbolId add_symbol(std::string name, SymbolKind kind);
    std::string_view symbol_name(SymbolId id) const { return symbols_[id].name; }
    SymbolKind symbol_kind(SymbolId id) const { return symbols_[id].kind; }

    const Atom& atom(AtomId id) const { return atoms_[id]; }

    Poly symbol(SymbolId id);
    Poly pack_size(SymbolId pack);

    // Integer floor division and remainder, folded when exact.
    Poly floor_div(const Poly& a, const Poly& b);
    Poly mod(const Poly& a, const Poly& b);
    Poly min(const Poly& a, const Poly& b);
    Poly max(const Poly& a, const Poly& b);

    // Replaces dimension symbols by polynomials and pack element counts by
    // the given polynomials, rebuilding every atom.
    struct Substitution {
        std::map<SymbolId, Poly> dims;
        std::map<SymbolId, Poly> pack_sizes;
    };
    Poly substitute(const Poly& poly, const Substitution& substitution);

    bool mentions(const Poly& poly, SymbolId symbol) const;

    // Source-like rendering, for diagnostics: `B * S`, `(D + 1) / 2`.
    std::string to_string(const Poly& poly) const;

private:
    struct Symbol {
        std::string name;
        SymbolKind kind;
    };
    using AtomKey = std::tuple<AtomKind, SymbolId, Poly, Poly>;

    AtomId intern(Atom atom);
    std::string atom_to_string(AtomId id) const;
    std::string operand_to_string(const Poly& poly) const;

    std::vector<Symbol> symbols_;
    std::vector<Atom> atoms_;
    std::map<AtomKey, AtomId> interned_;
};

} // namespace linnet::shape
