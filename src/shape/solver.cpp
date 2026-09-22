#include "linnet/shape/solver.hpp"

#include <algorithm>
#include <limits>

namespace linnet::shape {

namespace {

// Guards against runaway rewriting when equalities refer to each other.
constexpr int max_rewrite_depth = 8;

using Bound = std::optional<std::int64_t>;

Bound checked(std::int64_t a, std::int64_t b, bool is_product) {
    constexpr std::int64_t low = std::numeric_limits<std::int64_t>::min();
    constexpr std::int64_t high = std::numeric_limits<std::int64_t>::max();
    if (!is_product) {
        if ((b > 0 && a > high - b) || (b < 0 && a < low - b)) {
            return std::nullopt;
        }
        return a + b;
    }
    if (a == 0 || b == 0) {
        return 0;
    }
    // Operands of bound products are small in practice; refuse anything that
    // could overflow rather than reasoning about it.
    constexpr std::int64_t limit = std::int64_t{1} << 31;
    if (a <= -limit || a >= limit || b <= -limit || b >= limit) {
        return std::nullopt;
    }
    return a * b;
}

Bound add(Bound a, Bound b) {
    return a && b ? checked(*a, *b, false) : std::nullopt;
}

Bound multiply(Bound a, Bound b) {
    return a && b ? checked(*a, *b, true) : std::nullopt;
}

std::int64_t floor_modulo(std::int64_t a, std::int64_t b) {
    const std::int64_t remainder = a % b;
    return remainder < 0 ? remainder + b : remainder;
}

Poly term_poly(const Term& term) {
    Poly result(term.coefficient);
    for (const AtomId id : term.atoms) {
        result = result * Poly::atom(id);
    }
    return result;
}

} // namespace

// -------------------------------------------------------------- simplification

Poly Solver::simplify(const Poly& poly) {
    return simplify_at(poly, 0);
}

Poly Solver::simplify_at(const Poly& poly, int depth) {
    if (!poly.is_valid() || depth > max_rewrite_depth) {
        return poly;
    }
    Poly result;
    for (const Term& term : poly.terms()) {
        result = result + simplify_term(term, depth);
    }
    return result;
}

Poly Solver::simplify_term(const Term& term, int depth) {
    // The coefficient is a factor too, so that 2 * (D / 2) can become D.
    std::vector<Poly> factors{Poly(term.coefficient)};
    for (const AtomId id : term.atoms) {
        const Atom atom = context_->atom(id); // copy: interning may reallocate
        switch (atom.kind) {
        case AtomKind::Symbol: {
            const auto found = equalities_.find(atom.symbol);
            factors.push_back(found == equalities_.end() ? Poly::atom(id)
                                                         : simplify_at(found->second, depth + 1));
            break;
        }
        case AtomKind::PackSize:
            factors.push_back(Poly::atom(id));
            break;
        case AtomKind::FloorDiv: {
            Poly dividend = simplify_at(atom.lhs, depth + 1);
            const Poly divisor = simplify_at(atom.rhs, depth + 1);
            if (const auto constant = divisor.constant(); constant && *constant > 0) {
                // (m + r) / d == m / d when d divides m and 0 <= r < d.
                const Poly multiple =
                    dividend - Poly(floor_modulo(dividend.constant_term(), *constant));
                if (divides(divisor, multiple)) {
                    dividend = multiple;
                }
            }
            factors.push_back(context_->floor_div(dividend, divisor));
            break;
        }
        case AtomKind::Mod: {
            const Poly dividend = simplify_at(atom.lhs, depth + 1);
            const Poly divisor = simplify_at(atom.rhs, depth + 1);
            Poly value = context_->mod(dividend, divisor);
            if (divides(divisor, dividend)) {
                value = Poly(0);
            } else if (const auto constant = divisor.constant(); constant && *constant > 0) {
                const std::int64_t remainder = floor_modulo(dividend.constant_term(), *constant);
                if (divides(divisor, dividend - Poly(remainder))) {
                    value = Poly(remainder);
                }
            }
            factors.push_back(value);
            break;
        }
        case AtomKind::Min:
        case AtomKind::Max: {
            const Poly a = simplify_at(atom.lhs, depth + 1);
            const Poly b = simplify_at(atom.rhs, depth + 1);
            const bool is_min = atom.kind == AtomKind::Min;
            if (is_non_negative(b - a)) {
                factors.push_back(is_min ? a : b);
            } else if (is_non_negative(a - b)) {
                factors.push_back(is_min ? b : a);
            } else {
                factors.push_back(is_min ? context_->min(a, b) : context_->max(a, b));
            }
            break;
        }
        }
    }

    // (x / y) * y == x when y divides x.
    bool has_changed = true;
    while (has_changed) {
        has_changed = false;
        for (std::size_t i = 0; i < factors.size() && !has_changed; ++i) {
            const auto id = factors[i].single_atom();
            if (!id || context_->atom(*id).kind != AtomKind::FloorDiv) {
                continue;
            }
            const Atom quotient = context_->atom(*id);
            if (!quotient.rhs.is_single_term() || !divides(quotient.rhs, quotient.lhs)) {
                continue;
            }
            Poly others(1);
            for (std::size_t j = 0; j < factors.size(); ++j) {
                if (j != i) {
                    others = others * factors[j];
                }
            }
            if (const auto reduced = others.exact_div(quotient.rhs)) {
                factors = {*reduced, quotient.lhs};
                has_changed = true;
            }
        }
    }

    Poly result(1);
    for (const Poly& factor : factors) {
        result = result * factor;
    }
    return result;
}

bool Solver::divides(const Poly& divisor, const Poly& dividend) {
    if (!divisor.is_valid() || !dividend.is_valid()) {
        return false;
    }
    if (dividend.is_zero() || divisor == Poly(1) || divisor == Poly(-1)) {
        return true;
    }
    if (divisor.is_zero()) {
        return false;
    }
    for (const Divisibility& fact : divisibility_) {
        if (fact.divisor == divisor && fact.dividend == dividend) {
            return true;
        }
    }
    // Otherwise every term must be divisible on its own.
    for (const Term& term : dividend.terms()) {
        const Poly part = term_poly(term);
        bool is_divisible = part.exact_div(divisor).has_value();
        for (const Divisibility& fact : divisibility_) {
            if (is_divisible) {
                break;
            }
            is_divisible = fact.divisor == divisor && part.exact_div(fact.dividend).has_value();
        }
        if (!is_divisible) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------- bounds

Solver::Interval Solver::atom_bounds(AtomId id, int depth) {
    const Atom atom = context_->atom(id);
    switch (atom.kind) {
    case AtomKind::Symbol: {
        Interval interval{0, std::nullopt}; // dimensions are never negative
        if (const auto found = symbol_bounds_.find(atom.symbol); found != symbol_bounds_.end()) {
            if (found->second.low) {
                interval.low = std::max<std::int64_t>(0, *found->second.low);
            }
            interval.high = found->second.high;
        }
        return interval;
    }
    case AtomKind::PackSize:
        return {0, std::nullopt};
    case AtomKind::FloorDiv: {
        const Interval a = bounds(atom.lhs, depth + 1);
        const Interval b = bounds(atom.rhs, depth + 1);
        if (!a.low || *a.low < 0 || !b.low || *b.low < 1) {
            return {};
        }
        Interval interval{b.high ? Bound(*a.low / *b.high) : Bound(0), std::nullopt};
        if (a.high) {
            interval.high = *a.high / *b.low;
        }
        return interval;
    }
    case AtomKind::Mod: {
        const Interval b = bounds(atom.rhs, depth + 1);
        if (!b.low || *b.low < 1) {
            return {};
        }
        return {0, add(b.high, -1)};
    }
    case AtomKind::Min:
    case AtomKind::Max: {
        const Interval a = bounds(atom.lhs, depth + 1);
        const Interval b = bounds(atom.rhs, depth + 1);
        const auto pick = [](Bound x, Bound y, bool want_smaller, bool needs_both) -> Bound {
            if (x && y) {
                return want_smaller ? std::min(*x, *y) : std::max(*x, *y);
            }
            if (needs_both) {
                return std::nullopt;
            }
            return x ? x : y;
        };
        if (atom.kind == AtomKind::Min) {
            // min(a, b) <= each operand, and >= the smaller lower bound.
            return {pick(a.low, b.low, true, true), pick(a.high, b.high, true, false)};
        }
        return {pick(a.low, b.low, false, false), pick(a.high, b.high, false, true)};
    }
    }
    return {};
}

Solver::Interval Solver::bounds(const Poly& poly, int depth) {
    if (!poly.is_valid() || depth > max_rewrite_depth) {
        return {};
    }
    Interval total{0, 0};
    for (const Term& term : poly.terms()) {
        // Products are bounded only when every factor is known non-negative.
        Interval product{1, 1};
        for (const AtomId id : term.atoms) {
            const Interval factor = atom_bounds(id, depth);
            if (!factor.low || *factor.low < 0) {
                return {};
            }
            product.low = multiply(product.low, factor.low);
            const bool is_zero =
                (product.high && *product.high == 0) || (factor.high && *factor.high == 0);
            product.high = is_zero ? Bound(0) : multiply(product.high, factor.high);
        }
        const Bound coefficient = term.coefficient;
        const Bound scaled_low = multiply(coefficient, product.low);
        const Bound scaled_high = multiply(coefficient, product.high);
        total.low = add(total.low, term.coefficient > 0 ? scaled_low : scaled_high);
        total.high = add(total.high, term.coefficient > 0 ? scaled_high : scaled_low);
    }
    return total;
}

std::optional<std::int64_t> Solver::lower_bound(const Poly& poly) {
    return bounds(simplify(poly), 0).low;
}

std::optional<std::int64_t> Solver::upper_bound(const Poly& poly) {
    return bounds(simplify(poly), 0).high;
}

bool Solver::is_non_negative(const Poly& poly) {
    if (!poly.is_valid()) {
        return false;
    }
    const Bound low = bounds(poly, 0).low;
    if (low && *low >= 0) {
        return true;
    }
    // poly = fact + (poly - fact), with fact >= 0.
    const bool has_fact =
        std::any_of(non_negative_.begin(), non_negative_.end(), [&](const Poly& fact) {
            const Bound rest = bounds(poly - fact, 0).low;
            return rest && *rest >= 0;
        });
    if (has_fact) {
        return true;
    }
    // 0 <= x / c <= x for x >= 0 and a constant c >= 1, so replacing such a
    // quotient by x where it is subtracted, and by 0 where it is added, gives
    // a polynomial that is never larger than `poly`.
    Poly smaller = poly;
    bool has_changed = false;
    for (const Term& term : poly.terms()) {
        if (term.atoms.size() != 1) {
            continue;
        }
        const Atom atom = context_->atom(term.atoms.front());
        const auto divisor = atom.rhs.constant();
        if (atom.kind != AtomKind::FloorDiv || !divisor || *divisor < 1 ||
            !is_non_negative(atom.lhs)) {
            continue;
        }
        const Poly quotient = Poly::atom(term.atoms.front());
        const Poly replacement = term.coefficient < 0 ? atom.lhs : Poly(0);
        smaller = smaller + Poly(term.coefficient) * (replacement - quotient);
        has_changed = true;
    }
    if (!has_changed) {
        return false;
    }
    const Bound reduced = bounds(smaller, 0).low;
    return reduced && *reduced >= 0;
}

// --------------------------------------------------------------------- queries

bool Solver::prove(Relation relation, const Poly& lhs, const Poly& rhs) {
    const Poly difference = simplify(lhs - rhs);
    if (!difference.is_valid()) {
        return false;
    }
    switch (relation) {
    case Relation::Equal:
        return difference.is_zero() ||
               std::any_of(zero_.begin(), zero_.end(), [&](const Poly& fact) {
                   return fact == difference || fact == -difference;
               });
    case Relation::NotEqual:
        return is_non_negative(difference - Poly(1)) || is_non_negative(-difference - Poly(1));
    case Relation::Less:
        return is_non_negative(-difference - Poly(1));
    case Relation::LessEqual:
        return is_non_negative(-difference);
    case Relation::Greater:
        return is_non_negative(difference - Poly(1));
    case Relation::GreaterEqual:
        return is_non_negative(difference);
    }
    return false;
}

// ----------------------------------------------------------------- assumptions

void Solver::assume_non_negative(const Poly& poly) {
    if (!poly.is_valid()) {
        return;
    }
    // `S + k >= 0` and `-S + k >= 0` tighten the bounds of S directly.
    const Poly symbolic = poly - Poly(poly.constant_term());
    if (symbolic.is_single_term() && symbolic.terms().front().atoms.size() == 1) {
        const Term& term = symbolic.terms().front();
        const Atom& atom = context_->atom(term.atoms.front());
        if (atom.kind == AtomKind::Symbol && (term.coefficient == 1 || term.coefficient == -1)) {
            Interval& interval = symbol_bounds_[atom.symbol];
            const std::int64_t constant = poly.constant_term();
            if (term.coefficient == 1) {
                interval.low = std::max(interval.low.value_or(0), -constant);
            } else {
                interval.high = interval.high ? std::min(*interval.high, constant) : constant;
            }
            return;
        }
    }
    non_negative_.push_back(poly);
}

void Solver::assume(Relation relation, const Poly& lhs, const Poly& rhs) {
    const Poly left = simplify(lhs);
    const Poly right = simplify(rhs);
    const Poly difference = left - right;
    if (!difference.is_valid()) {
        return;
    }

    switch (relation) {
    case Relation::Equal: {
        for (const auto& [value, other] : {std::pair{left, right}, std::pair{right, left}}) {
            const auto id = value.single_atom();
            if (id && other.is_zero() && context_->atom(*id).kind == AtomKind::Mod) {
                const Atom atom = context_->atom(*id);
                divisibility_.push_back({atom.lhs, atom.rhs});
                return;
            }
        }
        // Solve for the most recently declared symbol that stands alone.
        std::optional<SymbolId> solved;
        Poly solution;
        for (const Term& term : difference.terms()) {
            if (term.atoms.size() != 1 || (term.coefficient != 1 && term.coefficient != -1)) {
                continue;
            }
            const Atom& atom = context_->atom(term.atoms.front());
            const Poly rest = difference - term_poly(term);
            if (atom.kind == AtomKind::Symbol && !context_->mentions(rest, atom.symbol) &&
                (!solved || atom.symbol > *solved)) {
                solved = atom.symbol;
                solution = term.coefficient == 1 ? -rest : rest;
            }
        }
        if (solved) {
            equalities_[*solved] = solution;
            for (Divisibility& fact : divisibility_) {
                fact = {simplify(fact.dividend), simplify(fact.divisor)};
            }
            for (Poly& fact : zero_) {
                fact = simplify(fact);
            }
            for (Poly& fact : non_negative_) {
                fact = simplify(fact);
            }
        } else if (!difference.is_zero()) {
            zero_.push_back(difference);
        }
        return;
    }
    case Relation::NotEqual:
        if (is_non_negative(difference)) {
            assume_non_negative(difference - Poly(1));
        } else if (is_non_negative(-difference)) {
            assume_non_negative(-difference - Poly(1));
        }
        return;
    case Relation::Less:
        assume_non_negative(-difference - Poly(1));
        return;
    case Relation::LessEqual:
        assume_non_negative(-difference);
        return;
    case Relation::Greater:
        assume_non_negative(difference - Poly(1));
        return;
    case Relation::GreaterEqual:
        assume_non_negative(difference);
        return;
    }
}

} // namespace linnet::shape
