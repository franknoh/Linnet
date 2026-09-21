#include "linnet/shape/dim.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace linnet::shape {

namespace {

constexpr std::int64_t int_min = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t int_max = std::numeric_limits<std::int64_t>::max();

std::optional<std::int64_t> checked_add(std::int64_t a, std::int64_t b) {
    if ((b > 0 && a > int_max - b) || (b < 0 && a < int_min - b)) {
        return std::nullopt;
    }
    return a + b;
}

std::optional<std::int64_t> checked_mul(std::int64_t a, std::int64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if ((a == -1 && b == int_min) || (b == -1 && a == int_min)) {
        return std::nullopt;
    }
    const std::int64_t product =
        static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
    if (product / b != a) {
        return std::nullopt;
    }
    return product;
}

std::int64_t floor_divide(std::int64_t a, std::int64_t b) {
    const std::int64_t quotient = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? quotient - 1 : quotient;
}

// Remainder with the sign of the (positive) divisor.
std::int64_t floor_modulo(std::int64_t a, std::int64_t b) {
    const std::int64_t remainder = a % b;
    return remainder < 0 ? remainder + b : remainder;
}

// Removes the atoms of `part` from `whole` when all are present.
std::optional<std::vector<AtomId>> remove_atoms(const std::vector<AtomId>& whole,
                                                const std::vector<AtomId>& part) {
    std::vector<AtomId> rest;
    std::size_t next = 0;
    for (const AtomId atom : whole) {
        if (next < part.size() && part[next] == atom) {
            ++next;
        } else {
            rest.push_back(atom);
        }
    }
    if (next != part.size()) {
        return std::nullopt;
    }
    return rest;
}

} // namespace

// ------------------------------------------------------------------------ Poly

Poly::Poly(std::int64_t constant) {
    if (constant != 0) {
        terms_.push_back({{}, constant});
    }
}

Poly Poly::invalid() {
    Poly poly;
    poly.is_valid_ = false;
    return poly;
}

Poly Poly::atom(AtomId id) {
    Poly poly;
    poly.terms_.push_back({{id}, 1});
    return poly;
}

std::optional<std::int64_t> Poly::constant() const {
    if (!is_valid_) {
        return std::nullopt;
    }
    if (terms_.empty()) {
        return 0;
    }
    if (terms_.size() == 1 && terms_.front().atoms.empty()) {
        return terms_.front().coefficient;
    }
    return std::nullopt;
}

std::int64_t Poly::constant_term() const {
    // The constant term has the empty atom list, which sorts first.
    if (!terms_.empty() && terms_.front().atoms.empty()) {
        return terms_.front().coefficient;
    }
    return 0;
}

std::optional<AtomId> Poly::single_atom() const {
    if (is_valid_ && terms_.size() == 1 && terms_.front().coefficient == 1 &&
        terms_.front().atoms.size() == 1) {
        return terms_.front().atoms.front();
    }
    return std::nullopt;
}

void Poly::normalize() {
    if (!is_valid_) {
        terms_.clear();
        return;
    }
    std::sort(terms_.begin(), terms_.end(), [](const Term& a, const Term& b) {
        return a.atoms < b.atoms;
    });
    std::vector<Term> merged;
    for (Term& term : terms_) {
        if (!merged.empty() && merged.back().atoms == term.atoms) {
            const auto sum = checked_add(merged.back().coefficient, term.coefficient);
            if (!sum) {
                *this = invalid();
                return;
            }
            merged.back().coefficient = *sum;
        } else {
            merged.push_back(std::move(term));
        }
    }
    std::erase_if(merged, [](const Term& term) { return term.coefficient == 0; });
    terms_ = std::move(merged);
}

Poly operator+(const Poly& a, const Poly& b) {
    if (!a.is_valid_ || !b.is_valid_) {
        return Poly::invalid();
    }
    Poly result = a;
    result.terms_.insert(result.terms_.end(), b.terms_.begin(), b.terms_.end());
    result.normalize();
    return result;
}

Poly Poly::operator-() const {
    return Poly(-1) * *this;
}

Poly operator-(const Poly& a, const Poly& b) {
    return a + (-b);
}

Poly operator*(const Poly& a, const Poly& b) {
    if (!a.is_valid_ || !b.is_valid_) {
        return Poly::invalid();
    }
    Poly result;
    for (const Term& x : a.terms_) {
        for (const Term& y : b.terms_) {
            const auto coefficient = checked_mul(x.coefficient, y.coefficient);
            if (!coefficient) {
                return Poly::invalid();
            }
            Term term{{}, *coefficient};
            term.atoms.resize(x.atoms.size() + y.atoms.size());
            std::merge(
                x.atoms.begin(), x.atoms.end(), y.atoms.begin(), y.atoms.end(), term.atoms.begin());
            result.terms_.push_back(std::move(term));
        }
    }
    result.normalize();
    return result;
}

std::optional<Poly> Poly::exact_div(const Poly& divisor) const {
    if (!is_valid_ || !divisor.is_single_term()) {
        return std::nullopt;
    }
    const Term& by = divisor.terms_.front();
    Poly result;
    for (const Term& term : terms_) {
        if (term.coefficient % by.coefficient != 0 ||
            (term.coefficient == int_min && by.coefficient == -1)) {
            return std::nullopt;
        }
        auto atoms = remove_atoms(term.atoms, by.atoms);
        if (!atoms) {
            return std::nullopt;
        }
        result.terms_.push_back({std::move(*atoms), term.coefficient / by.coefficient});
    }
    result.normalize();
    return result;
}

// ------------------------------------------------------------------ DimContext

SymbolId DimContext::add_symbol(std::string name, SymbolKind kind) {
    symbols_.push_back({std::move(name), kind});
    return static_cast<SymbolId>(symbols_.size() - 1);
}

AtomId DimContext::intern(Atom atom) {
    AtomKey key{atom.kind, atom.symbol, atom.lhs, atom.rhs};
    const auto found = interned_.find(key);
    if (found != interned_.end()) {
        return found->second;
    }
    atoms_.push_back(std::move(atom));
    const auto id = static_cast<AtomId>(atoms_.size() - 1);
    interned_.emplace(std::move(key), id);
    return id;
}

Poly DimContext::symbol(SymbolId id) {
    return Poly::atom(intern({AtomKind::Symbol, id, {}, {}}));
}

Poly DimContext::pack_size(SymbolId pack) {
    return Poly::atom(intern({AtomKind::PackSize, pack, {}, {}}));
}

Poly DimContext::floor_div(const Poly& a, const Poly& b) {
    if (!a.is_valid() || !b.is_valid()) {
        return Poly::invalid();
    }
    if (const auto divisor = b.constant()) {
        if (*divisor == 0) {
            return Poly::invalid();
        }
        if (const auto dividend = a.constant()) {
            if (*dividend == int_min && *divisor == -1) {
                return Poly::invalid();
            }
            return Poly(floor_divide(*dividend, *divisor));
        }
        if (*divisor > 0) {
            // a = multiple + r with 0 <= r < divisor, so floor(a / divisor)
            // is multiple / divisor whenever that quotient is exact.
            const Poly multiple = a - Poly(floor_modulo(a.constant_term(), *divisor));
            if (auto quotient = multiple.exact_div(b)) {
                return *quotient;
            }
        }
    } else if (a.is_zero()) {
        return a;
    } else if (auto quotient = a.exact_div(b)) {
        return *quotient;
    }
    return Poly::atom(intern({AtomKind::FloorDiv, 0, a, b}));
}

Poly DimContext::mod(const Poly& a, const Poly& b) {
    if (!a.is_valid() || !b.is_valid()) {
        return Poly::invalid();
    }
    if (const auto divisor = b.constant()) {
        if (*divisor == 0) {
            return Poly::invalid();
        }
        if (*divisor > 0) {
            const std::int64_t remainder = floor_modulo(a.constant_term(), *divisor);
            if ((a - Poly(remainder)).exact_div(b)) {
                return Poly(remainder);
            }
        }
    } else if (a.is_zero() || a.exact_div(b)) {
        return Poly(0);
    }
    return Poly::atom(intern({AtomKind::Mod, 0, a, b}));
}

Poly DimContext::min(const Poly& a, const Poly& b) {
    if (!a.is_valid() || !b.is_valid()) {
        return Poly::invalid();
    }
    if (a == b) {
        return a;
    }
    const auto x = a.constant();
    const auto y = b.constant();
    if (x && y) {
        return Poly(std::min(*x, *y));
    }
    return Poly::atom(intern({AtomKind::Min, 0, std::min(a, b), std::max(a, b)}));
}

Poly DimContext::max(const Poly& a, const Poly& b) {
    if (!a.is_valid() || !b.is_valid()) {
        return Poly::invalid();
    }
    if (a == b) {
        return a;
    }
    const auto x = a.constant();
    const auto y = b.constant();
    if (x && y) {
        return Poly(std::max(*x, *y));
    }
    return Poly::atom(intern({AtomKind::Max, 0, std::min(a, b), std::max(a, b)}));
}

Poly DimContext::substitute(const Poly& poly, const Substitution& substitution) {
    if (!poly.is_valid()) {
        return poly;
    }
    Poly result;
    for (const Term& term : poly.terms()) {
        Poly product(term.coefficient);
        for (const AtomId id : term.atoms) {
            const Atom source = atom(id); // copy: interning below may reallocate
            Poly value = Poly::atom(id);
            switch (source.kind) {
            case AtomKind::Symbol:
                if (const auto found = substitution.dims.find(source.symbol);
                    found != substitution.dims.end()) {
                    value = found->second;
                }
                break;
            case AtomKind::PackSize:
                if (const auto found = substitution.pack_sizes.find(source.symbol);
                    found != substitution.pack_sizes.end()) {
                    value = found->second;
                }
                break;
            case AtomKind::FloorDiv:
                value = floor_div(substitute(source.lhs, substitution),
                                  substitute(source.rhs, substitution));
                break;
            case AtomKind::Mod:
                value =
                    mod(substitute(source.lhs, substitution), substitute(source.rhs, substitution));
                break;
            case AtomKind::Min:
                value =
                    min(substitute(source.lhs, substitution), substitute(source.rhs, substitution));
                break;
            case AtomKind::Max:
                value =
                    max(substitute(source.lhs, substitution), substitute(source.rhs, substitution));
                break;
            }
            product = product * value;
        }
        result = result + product;
    }
    return result;
}

bool DimContext::mentions(const Poly& poly, SymbolId symbol) const {
    for (const Term& term : poly.terms()) {
        for (const AtomId id : term.atoms) {
            const Atom& candidate = atom(id);
            if (candidate.kind == AtomKind::Symbol || candidate.kind == AtomKind::PackSize) {
                if (candidate.symbol == symbol) {
                    return true;
                }
            } else if (mentions(candidate.lhs, symbol) || mentions(candidate.rhs, symbol)) {
                return true;
            }
        }
    }
    return false;
}

std::string DimContext::operand_to_string(const Poly& poly) const {
    const bool is_simple_constant = poly.constant() && *poly.constant() >= 0;
    bool is_simple_atom = false;
    if (const auto id = poly.single_atom()) {
        const AtomKind kind = atom(*id).kind;
        is_simple_atom = kind != AtomKind::FloorDiv && kind != AtomKind::Mod;
    }
    const std::string text = to_string(poly);
    return is_simple_constant || is_simple_atom ? text : "(" + text + ")";
}

std::string DimContext::atom_to_string(AtomId id) const {
    const Atom& value = atom(id);
    switch (value.kind) {
    case AtomKind::Symbol:
        return std::string(symbol_name(value.symbol));
    case AtomKind::PackSize:
        return "numel(*" + std::string(symbol_name(value.symbol)) + ")";
    case AtomKind::FloorDiv:
        return operand_to_string(value.lhs) + " / " + operand_to_string(value.rhs);
    case AtomKind::Mod:
        return operand_to_string(value.lhs) + " % " + operand_to_string(value.rhs);
    case AtomKind::Min:
        return "min(" + to_string(value.lhs) + ", " + to_string(value.rhs) + ")";
    case AtomKind::Max:
        return "max(" + to_string(value.lhs) + ", " + to_string(value.rhs) + ")";
    }
    return "?";
}

std::string DimContext::to_string(const Poly& poly) const {
    if (!poly.is_valid()) {
        return "<overflow>";
    }
    if (poly.is_zero()) {
        return "0";
    }
    // Symbolic terms first, the constant last: `D + 1` rather than `1 + D`.
    std::vector<const Term*> ordered;
    for (const Term& term : poly.terms()) {
        if (!term.atoms.empty()) {
            ordered.push_back(&term);
        }
    }
    for (const Term& term : poly.terms()) {
        if (term.atoms.empty()) {
            ordered.push_back(&term);
        }
    }

    std::string text;
    for (const Term* term : ordered) {
        const bool is_negative = term->coefficient < 0;
        if (text.empty()) {
            text += is_negative ? "-" : "";
        } else {
            text += is_negative ? " - " : " + ";
        }
        // Avoid negating the minimum value; print its digits directly.
        std::string magnitude = std::to_string(term->coefficient);
        if (is_negative) {
            magnitude.erase(0, 1);
        }
        const bool has_factor = magnitude != "1" || term->atoms.empty();
        if (has_factor) {
            text += magnitude;
        }
        const bool is_product = term->atoms.size() + (has_factor ? 1 : 0) > 1;
        for (std::size_t i = 0; i < term->atoms.size(); ++i) {
            if (i != 0 || has_factor) {
                text += " * ";
            }
            const AtomKind kind = atom(term->atoms[i]).kind;
            const bool needs_parens =
                is_product && (kind == AtomKind::FloorDiv || kind == AtomKind::Mod);
            const std::string factor = atom_to_string(term->atoms[i]);
            text += needs_parens ? "(" + factor + ")" : factor;
        }
    }
    return text;
}

} // namespace linnet::shape
