#pragma once

#include "linnet/shape/dim.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace linnet::shape {

enum class Relation : std::uint8_t { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };

// Proves relations between dimension expressions from declared constraints.
//
// The solver is sound and deliberately incomplete: `prove` returning false
// means "not proven", never "false". It reasons with
//   - polynomial normalization (from Poly),
//   - substitution of symbols fixed by equality constraints,
//   - divisibility facts (`H % N == 0`), which let `(H / N) * N` become `H`,
//   - interval bounds, knowing every dimension symbol is non-negative.
class Solver {
public:
    explicit Solver(DimContext& context) : context_(&context) {}

    // Records a constraint as an assumption. Constraints the solver cannot
    // use are ignored, which is always sound.
    void assume(Relation relation, const Poly& lhs, const Poly& rhs);

    bool prove(Relation relation, const Poly& lhs, const Poly& rhs);
    bool prove_equal(const Poly& a, const Poly& b) { return prove(Relation::Equal, a, b); }

    // Rewrites `poly` using the recorded equalities and divisibility facts.
    Poly simplify(const Poly& poly);

    // Whether `divisor` provably divides `dividend`.
    bool divides(const Poly& divisor, const Poly& dividend);

    // Inclusive bounds implied by the assumptions; nullopt is unbounded.
    std::optional<std::int64_t> lower_bound(const Poly& poly);
    std::optional<std::int64_t> upper_bound(const Poly& poly);

private:
    struct Divisibility {
        Poly dividend;
        Poly divisor;
    };
    struct Interval {
        std::optional<std::int64_t> low;
        std::optional<std::int64_t> high;
    };

    void assume_non_negative(const Poly& poly);
    Poly simplify_term(const Term& term, int depth);
    Poly simplify_at(const Poly& poly, int depth);
    Interval bounds(const Poly& poly, int depth);
    Interval atom_bounds(AtomId id, int depth);
    bool is_non_negative(const Poly& poly);

    DimContext* context_;
    std::map<SymbolId, Poly> equalities_;
    std::vector<Divisibility> divisibility_;
    std::vector<Poly> zero_;         // polynomials known to equal zero
    std::vector<Poly> non_negative_; // polynomials known to be >= 0
    std::map<SymbolId, Interval> symbol_bounds_;
};

} // namespace linnet::shape
